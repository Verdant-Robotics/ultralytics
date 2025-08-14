# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
import numpy as np
import torch

from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, box_iou, kpt_iou
from ultralytics.utils.plotting import output_to_target, plot_images
from ultralytics.utils.tal import make_anchors


class PoseSegValidator(PoseValidator):

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize a 'PoseValidator' object with custom parameters and assigned attributes."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = 'pose-segmentation'

    def init_metrics(self, model):
        super().init_metrics(model)
        self.seg_ch_num = self.model.seg_ch_num

    def process_seg_result(self, preds):
        """
        Input: 
            preds = x_flat, ([P1, P2, P3], kpt)
            Each Pi is (bs, self.no, h_i, w_i), with h_i and w_i being different for each P. e.g 8x8, 4x4, 2x2 corresponding to resolution(self.stride)
            x_flat = (bs, xyxy(bbox),cls0,..,clsi,seg_obj, seg0,...,segj, A) e.g A = anchors_len = 8x8 + 4x4 + 2x2 = 84
        Output:
            seg_logit (B, seg_ch_num, A)
            Pi_list for reconstructing anchors later on
        """
        Pi_list = preds[1][0]
        x_flat = preds[0]
        seg_offset = 4 + self.nc
        seg_obj_sh = x_flat[:, seg_offset : seg_offset + 1, :]
        seg_obj_unsh = x_flat[:, seg_offset + 1 : seg_offset + 2, :]
        seg_logits = x_flat[:, seg_offset + 2 : seg_offset + 2 + self.seg_ch_num, :]
        return seg_obj_unsh, seg_logits, Pi_list


    def postprocess(self, preds):
        """
        Apply non-maximum suppression and return detections with high confidence scores + Map anchor_points to seg classes
        """
        return ops.non_max_suppression(prediction=preds,
                                       nc=self.nc,
                                       conf_thres=self.args.conf,
                                       iou_thres=self.args.iou,
                                       labels=self.lb,
                                       multi_label=True,
                                       agnostic=self.args.single_cls,
                                       max_det=self.args.max_det), self.process_seg_result(preds)

    def update_metrics(self, preds, batch):
        """This is copied from PoseValidator. For unknown reasons, it will not work if we don't copy it here."""
        preds = preds[0]
        for si, pred in enumerate(preds):
            idx = batch['batch_idx'] == si
            cls = batch['cls'][idx].clamp(max=self.nc - 1)
            bbox = batch['bboxes'][idx]
            kpts = batch['keypoints'][idx]
            nl, npr = cls.shape[0], pred.shape[0]  # number of labels, predictions
            nk = kpts.shape[1]  # number of keypoints
            shape = batch['ori_shape'][si]
            correct_kpts = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            correct_bboxes = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct_bboxes, correct_kpts, *torch.zeros(
                        (2, 0), device=self.device), cls.squeeze(-1)))
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, labels=cls.squeeze(-1))
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            ops.scale_boxes(batch['img'][si].shape[1:], predn[:, :4], shape,
                            ratio_pad=batch['ratio_pad'][si])  # native-space pred
            pred_kpts = predn[:, 6:].view(npr, nk, -1)
            ops.scale_coords(batch['img'][si].shape[1:], pred_kpts, shape, ratio_pad=batch['ratio_pad'][si])

            # Evaluate
            if nl:
                height, width = batch['img'].shape[2:]
                tbox = ops.xywh2xyxy(bbox) * torch.tensor(
                    (width, height, width, height), device=self.device)  # target boxes
                ops.scale_boxes(batch['img'][si].shape[1:], tbox, shape,
                                ratio_pad=batch['ratio_pad'][si])  # native-space labels
                tkpts = kpts.clone()
                tkpts[..., 0] *= width
                tkpts[..., 1] *= height
                tkpts = ops.scale_coords(batch['img'][si].shape[1:], tkpts, shape, ratio_pad=batch['ratio_pad'][si])
                labelsn = torch.cat((cls, tbox), 1)  # native-space labels
                correct_bboxes = self._process_batch(predn[:, :6], labelsn)
                correct_kpts = self._process_batch(predn[:, :6], labelsn, pred_kpts, tkpts)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, labelsn)

            # Append correct_masks, correct_boxes, pconf, pcls, tcls
            self.stats.append((correct_bboxes, correct_kpts, pred[:, 4], pred[:, 5], cls.squeeze(-1)))

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch['im_file'][si])
            # if self.args.save_txt:
            #    save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')

    def get_anchors_and_strides(self, Pi_list):
        '''
        Maps each anchor to its corresponding segmentation class.
        Input:
            seg_logit (B, seg_ch_num, A) - logits for each segmentation class for each anchor
            Pi_list feature list used to construct the anchor points
        Output:
            seg_results (seg_logits, anchor_points, strides) with shapes (B, C, A), (B, 2, A), (B, 1, A)
        '''
        stride_tensor = self.model.stride
        if type(stride_tensor) is int: # When models are loaded, model.stride is the max stride
            max_stride = stride_tensor 
            stride_tensor = torch.tensor([max_stride/4, max_stride/2, max_stride])

        anchor_points, strides = (x.transpose(0, 1) for x in make_anchors(Pi_list, stride_tensor, 0.5))
        bs = Pi_list[0].shape[0]
        strides = strides.unsqueeze(0).repeat(bs, 1, 1)
        anchor_points = anchor_points.unsqueeze(0).repeat(bs, 1, 1)
        return (anchor_points, strides)


    def plot_predictions(self, batch, predictions, ni):
        pred_bbox_kpts = predictions[0]
        kpt_offset = 4 + self.nc + self.seg_ch_num + 1 + 1 # xyxy + C + S + 2(seg_obj for sh + unsh)
        pred_kpts = torch.cat([p[:, kpt_offset:].view(-1, *self.kpt_shape) for p in pred_bbox_kpts], 0)
        batch_idx, cls, bboxes = output_to_target(pred_bbox_kpts, max_det=self.args.max_det)
        pred_seg_obj, pred_seg_clsfy, Pi_list  = predictions[1]

        anchors, strides = self.get_anchors_and_strides(Pi_list=Pi_list)
        pred_seg = pred_seg_clsfy * (pred_seg_obj > 0.4).float()
        seg_results = (pred_seg, anchors, strides)

        # plot bbox and kpts
        plot_images(images=batch['img'],
                    batch_idx=batch_idx,
                    cls=cls,
                    bboxes=bboxes,
                    kpts=pred_kpts,
                    fname=self.save_dir / f'val_batch{ni}_pred_bbox_kpt.jpg',
                    names=self.names,
                    on_plot=self.on_plot,
                    )

        # plot seg results for one resolution
        plot_images(images=batch['img'],
                    batch_idx=batch_idx,
                    cls=cls,
                    fname=self.save_dir / f'val_batch{ni}_pred_seg_grid8.jpg',
                    names=self.names,
                    on_plot=self.on_plot,
                    seg_results=seg_results,
                    res_grid_size=[8],
                    enable_highlight=True
                    )
