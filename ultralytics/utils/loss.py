# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.metrics import OKS_SIGMA
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import TaskAlignedAssigner, dist2bbox, make_anchors

from .metrics import bbox_iou
from .tal import bbox2dist

import itertools


class VarifocalLoss(nn.Module):
    """
    Varifocal loss by Zhang et al.

    https://arxiv.org/abs/2008.13367.
    """

    def __init__(self):
        """Initialize the VarifocalLoss class."""
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        """Computes varfocal loss."""
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction='none') *
                    weight).mean(1).sum()
        return loss


class FocalLoss(nn.Module):
    """Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)."""

    def __init__(self, ):
        """Initializer for FocalLoss class with no parameters."""
        super().__init__()

    @staticmethod
    def forward(pred, label, gamma=1.5, alpha=0.25):
        """Calculates and updates confusion matrix for object detection/classification tasks."""
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor
        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor
        return loss.mean(1).sum()


class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""

    def __init__(self, reg_max, use_dfl=False):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        """IoU loss."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.use_dfl:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            loss_dfl = self._df_loss(pred_dist[fg_mask].view(-1, self.reg_max + 1), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl

    @staticmethod
    def _df_loss(pred_dist, target):
        """Return sum of left and right DFL losses."""
        # Distribution Focal Loss (DFL) proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        return (F.cross_entropy(pred_dist, tl.view(-1), reduction='none').view(tl.shape) * wl +
                F.cross_entropy(pred_dist, tr.view(-1), reduction='none').view(tl.shape) * wr).mean(-1, keepdim=True)


class KeypointLoss(nn.Module):
    """Criterion class for computing training losses."""

    def __init__(self, sigmas) -> None:
        """Initialize the KeypointLoss class."""
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        """Calculates keypoint loss factor and Euclidean distance loss for predicted and actual keypoints."""
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]) ** 2 + (pred_kpts[..., 1] - gt_kpts[..., 1]) ** 2
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        # e = d / (2 * (area * self.sigmas) ** 2 + 1e-9)  # from formula
        e = d / (2 * self.sigmas) ** 2 / (area + 1e-9) / 2  # from cocoeval
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """Criterion class for computing training losses."""
    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters

        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device

        self.use_dfl = m.reg_max > 1

        self.assigner = TaskAlignedAssigner(topk=10, num_classes=self.nc, alpha=0.5, beta=6.0)
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=self.use_dfl).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted object bounding box coordinates from anchor points and distribution."""
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def __call__(self, preds, batch):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat((batch['batch_idx'].view(-1, 1), batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes the v8SegmentationLoss class, taking a de-paralleled model as argument."""
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        """Calculate and return the loss for the YOLO model."""
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape  # batch size, number of masks, mask height, mask width
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        try:
            batch_idx = batch['batch_idx'].view(-1, 1)
            targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)
        except RuntimeError as e:
            raise TypeError('ERROR ❌ segment dataset incorrectly formatted or not a segment dataset.\n'
                            "This error can occur when incorrectly training a 'segment' model on a 'detect' dataset, "
                            "i.e. 'yolo train model=yolov8n-seg.pt data=coco128.yaml'.\nVerify your dataset is a "
                            "correctly formatted 'segment' dataset using 'data=coco128-seg.yaml' "
                            'as an example.\nSee https://docs.ultralytics.com/tasks/segment/ for help.') from e

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        if fg_mask.sum():
            # Bbox loss
            loss[0], loss[3] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
                                              target_scores, target_scores_sum, fg_mask)
            # Masks loss
            masks = batch['masks'].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):  # downsample
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode='nearest')[0]

            loss[1] = self.calculate_segmentation_loss(fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto,
                                                       pred_masks, imgsz, self.overlap)

        # WARNING: lines below prevent Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.box  # seg gain
        loss[2] *= self.hyp.cls  # cls gain
        loss[3] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def single_mask_loss(gt_mask: torch.Tensor, pred: torch.Tensor, proto: torch.Tensor, xyxy: torch.Tensor,
                         area: torch.Tensor) -> torch.Tensor:
        """
        Compute the instance segmentation loss for a single image.

        Args:
            gt_mask (torch.Tensor): Ground truth mask of shape (n, H, W), where n is the number of objects.
            pred (torch.Tensor): Predicted mask coefficients of shape (n, 32).
            proto (torch.Tensor): Prototype masks of shape (32, H, W).
            xyxy (torch.Tensor): Ground truth bounding boxes in xyxy format, normalized to [0, 1], of shape (n, 4).
            area (torch.Tensor): Area of each ground truth bounding box of shape (n,).

        Returns:
            (torch.Tensor): The calculated mask loss for a single image.

        Notes:
            The function uses the equation pred_mask = torch.einsum('in,nhw->ihw', pred, proto) to produce the
            predicted masks from the prototype masks and predicted mask coefficients.
        """
        pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (n, 32) @ (32, 80, 80) -> (n, 80, 80)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(
        self,
        fg_mask: torch.Tensor,
        masks: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes: torch.Tensor,
        batch_idx: torch.Tensor,
        proto: torch.Tensor,
        pred_masks: torch.Tensor,
        imgsz: torch.Tensor,
        overlap: bool,
    ) -> torch.Tensor:
        """
        Calculate the loss for instance segmentation.

        Args:
            fg_mask (torch.Tensor): A binary tensor of shape (BS, N_anchors) indicating which anchors are positive.
            masks (torch.Tensor): Ground truth masks of shape (BS, H, W) if `overlap` is False, otherwise (BS, ?, H, W).
            target_gt_idx (torch.Tensor): Indexes of ground truth objects for each anchor of shape (BS, N_anchors).
            target_bboxes (torch.Tensor): Ground truth bounding boxes for each anchor of shape (BS, N_anchors, 4).
            batch_idx (torch.Tensor): Batch indices of shape (N_labels_in_batch, 1).
            proto (torch.Tensor): Prototype masks of shape (BS, 32, H, W).
            pred_masks (torch.Tensor): Predicted masks for each anchor of shape (BS, N_anchors, 32).
            imgsz (torch.Tensor): Size of the input image as a tensor of shape (2), i.e., (H, W).
            overlap (bool): Whether the masks in `masks` tensor overlap.

        Returns:
            (torch.Tensor): The calculated loss for instance segmentation.

        Notes:
            The batch loss can be computed for improved speed at higher memory usage.
            For example, pred_mask can be computed as follows:
                pred_mask = torch.einsum('in,nhw->ihw', pred, proto)  # (i, 32) @ (32, 160, 160) -> (i, 160, 160)
        """
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        # Normalize to 0-1
        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]

        # Areas of target bboxes
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)

        # Normalize to mask size
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i],
                                              marea_i[fg_mask_i])

            # WARNING: lines below prevents Multi-GPU DDP 'unused gradient' PyTorch errors, do not remove
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()  # inf sums may lead to nan loss

        return loss / fg_mask.sum()


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]  # number of keypoints
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # loss[1] = self.varifocal_loss(pred_scores, target_scores, target_labels) / target_scores_sum  # VFL way
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts, batch['ignore_kpt'])

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        """Decodes predicted keypoints to image coordinates."""
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes,
                                 pred_kpts, ignore_kpt):
        """
        Calculate the keypoints loss for the model.

        This function calculates the keypoints loss and keypoints object loss for a given batch. The keypoints loss is
        based on the difference between the predicted keypoints and ground truth keypoints. The keypoints object loss is
        a binary classification loss that classifies whether a keypoint is present or not.

        Args:
            masks (torch.Tensor): Binary mask tensor indicating object presence, shape (BS, N_anchors).
            target_gt_idx (torch.Tensor): Index tensor mapping anchors to ground truth objects, shape (BS, N_anchors).
            keypoints (torch.Tensor): Ground truth keypoints, shape (N_kpts_in_batch, N_kpts_per_object, kpts_dim).
            batch_idx (torch.Tensor): Batch index tensor for keypoints, shape (N_kpts_in_batch, 1).
            stride_tensor (torch.Tensor): Stride tensor for anchors, shape (N_anchors, 1).
            target_bboxes (torch.Tensor): Ground truth boxes in (x1, y1, x2, y2) format, shape (BS, N_anchors, 4).
            pred_kpts (torch.Tensor): Predicted keypoints, shape (BS, N_anchors, N_kpts_per_object, kpts_dim).

        Returns:
            (tuple): Returns a tuple containing:
                - kpts_loss (torch.Tensor): The keypoints loss.
                - kpts_obj_loss (torch.Tensor): The keypoints object loss.
        """
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        # Find the maximum number of keypoints in a single image
        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        # Create a tensor to hold batched keypoints
        batched_keypoints = torch.zeros((batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]),
                                        device=keypoints.device)

        # TODO: any idea how to vectorize this?
        # Fill batched_keypoints with keypoints based on batch_idx
        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, :keypoints_i.shape[0]] = keypoints_i

        # Expand dimensions of target_gt_idx to match the shape of batched_keypoints
        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        # Use target_gt_idx_expanded to select keypoints from batched_keypoints
        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2]))

        # Divide coordinates by stride
        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        for i in range(batch_size):
            if ignore_kpt[i]:
                # We should ignore the keypoints for this image
                # We replace gt keypoints with pred keypoints so distance is 0. kpts_loss will be 0
                selected_keypoints[i] = pred_kpts[i]

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)  # pose loss

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())  # keypoint obj loss

        return kpts_loss, kpts_obj_loss


class v8PoseSegLoss(v8PoseLoss):

    def __init__(self, model):
        super().__init__(model)
        self.bce_inside = nn.BCEWithLogitsLoss(reduction='none')
        self.seg_ch_num = model.model[-1].seg_ch_num
        self.training = model.training

    def __call__(self, preds, batch):
        loss = torch.zeros(6, device=self.device) # box, cls, dfl, kpt_location, kpt_visibility, segmentation
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        all_preds = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores, pred_seg = all_preds.split((self.reg_max * 4, self.nc, self.seg_ch_num), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        pred_seg = pred_seg.permute(0, 2, 1).contiguous()  # permute to (B, A, S)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        batch_size = pred_scores.shape[0]
        gt_labels, gt_bboxes, gt_bboxes_img = self.get_gt_targets(batch, batch_size, imgsz)

        loss = self.calculate_loss_for_none_shuffled_parts(loss, batch, gt_labels, gt_bboxes, pred_scores, pred_kpts, pred_distri, feats, imgsz)
        loss[5] = self.calculate_segmentation_loss(pred_seg=pred_seg, gt_bboxes_img=gt_bboxes_img)

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain
        loss[5] *= self.hyp.seg  # seg gain
        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


    def get_gt_targets(self, batch, batch_size, imgsz):
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2) # (B, T, 1), (B, T, 4)

        bboxes_img = batch['bboxes_img'].float() # B, C, H, W
        bboxes_img_multi_res = [F.interpolate(bboxes_img,
                                    scale_factor=float(1/stride),
                                    mode='nearest').flatten(start_dim=2) for stride in self.stride]
        gt_bboxes_img = torch.cat(bboxes_img_multi_res, dim=2).to(self.device)
        B, C, L = gt_bboxes_img.shape 
        if C < self.seg_ch_num:
            pad = self.seg_ch_num - C
            gt_bboxes_img = torch.cat(
                [gt_bboxes_img, torch.zeros(B, pad, L, device=self.device, dtype=gt_bboxes_img.dtype)],
                dim=1
            )
        return gt_labels, gt_bboxes, gt_bboxes_img


    def calculate_segmentation_loss(self, pred_seg, gt_bboxes_img):
        """
            pred_seg (B, A, seg_ch_num)
            bboxes_img (B, C, A)
        """
        # TODO: Should collapse bboxes_img if needed depending on C vs seg_ch_num
        target_seg = gt_bboxes_img.permute(0, 2, 1) # B, A, C
        loss_per_anchor = self.bce_inside(pred_seg, target_seg)  # (B, A, C)
        anchor_weights = torch.ones_like(loss_per_anchor)  # (B, A, C)
        weighted_loss = loss_per_anchor * anchor_weights # (B, A, C)
        return weighted_loss.mean()


    def calculate_loss_for_none_shuffled_parts(self, loss, batch, gt_labels, gt_bboxes, pred_scores, pred_kpts, pred_distri, feats, imgsz):
        not_shuffled_mask = ~batch['is_shuffled'].squeeze(1).to(self.device)
    
        gt_bboxes_ns = gt_bboxes[not_shuffled_mask]
        gt_labels_ns = gt_labels[not_shuffled_mask]
        mask_gt_ns = gt_bboxes_ns.sum(2, keepdim=True).gt_(0)

        feats_ns = [feat[not_shuffled_mask] for feat in feats ]
        pred_distri_ns = pred_distri[not_shuffled_mask]
        pred_kpts_ns = pred_kpts[not_shuffled_mask]
        pred_scores_ns = pred_scores[not_shuffled_mask]
        batch_size = pred_scores_ns.shape[0]
        dtype = pred_scores_ns.dtype

        anchor_points, stride_tensor = make_anchors(feats_ns, self.stride, 0.5)
        pred_bboxes_ns = self.bbox_decode(anchor_points, pred_distri_ns)  # (B, h x w, 4(xyxy))
        pred_kpts_ns = self.kpts_decode(anchor_points, pred_kpts_ns.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes_ns, target_scores_ns, fg_mask_ns, target_gt_idx_ns = self.assigner(
            pred_scores_ns.detach().sigmoid(), (pred_bboxes_ns.detach() * stride_tensor).type(gt_bboxes_ns.dtype),
            anchor_points * stride_tensor, gt_labels_ns, gt_bboxes_ns, mask_gt_ns)

        target_scores_sum_ns = max(target_scores_ns.sum(), 1)

        loss[3] = self.bce(pred_scores_ns, target_scores_ns.to(dtype)).sum() / target_scores_sum_ns

        if fg_mask_ns.sum(): # if any anchor has gt
            target_bboxes_ns /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri_ns, pred_bboxes_ns, anchor_points, target_bboxes_ns, target_scores_ns,
                                              target_scores_sum_ns, fg_mask_ns)

            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]
            
            batch_idx = batch['batch_idx'].view(-1, 1).to(self.device)
            flat_batch_idx = batch_idx.view(-1).long()
            mask = not_shuffled_mask[flat_batch_idx]

            batch_idx_ns = batch_idx[mask]
            keypoints_ns = keypoints[mask]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask_ns, target_gt_idx_ns, keypoints_ns, batch_idx_ns,
                                                             stride_tensor, target_bboxes_ns, pred_kpts_ns, batch['ignore_kpt'])
        return loss


class v8PoseTunableHeadLoss(v8PoseLoss):
    """Criterion class for computing training losses."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseTunableHeadLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(5, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls (B, n_anchors, 1), xyxy (B, n_anchors, 4)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # boolean vector indicating exitance of gt bbox (B, n_anchors, 1)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        # find index of image that has no gt_labes (nothing detected) and make bce loss for that index 0
        loss[3] = torch.zeros((), device=self.device, dtype=dtype, requires_grad=True) 
        for i, mask in enumerate(mask_gt):  # iterate over batch dim
            cls_loss = self.bce(pred_scores[i], target_scores[i].to(dtype)).sum()
            loss[3] += cls_loss if mask.sum() > 0 else cls_loss * 0.0  # don't update model if no gt bbox
        loss[3] /= target_scores_sum

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts, batch['ignore_kpt'])

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)
    

class v8PoseContrastiveLoss(v8PoseLoss):
    """Criterion class for computing training losses."""
    def __init__(self, model):
        """Initializes the v8PoseContrastiveLoss class."""
        super().__init__(model)
        self.n_samples = 10  # number of negative samples

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(7, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility, triplet loss, avg_logits
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]

        pred_distri, pred_scores, embeddings = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc, self.no - (self.reg_max * 4 + self.nc)), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()
        embeddings = embeddings.permute(0, 2, 1).contiguous()  # (B, n_features, n_anchors) -> (B, n_anchors, n_features)

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # Cls loss
        # # Modify gt labels to have "0" labels for classification.
        # # BCE loss will do binary classification per each class (predicted label vs none) per detection.
        # # This is equivalent to Plant vs. None (background) classification.
        # # This minimizes classification’s impact on generating embeddings 
        # # while triplet loss is encouraging the model to distinguish one class from another.
        label_for_single_class_cls = 0
        gt_labels_single_class_cls = torch.full_like(gt_labels, label_for_single_class_cls)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels_single_class_cls, gt_bboxes, mask_gt)

        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask_ns, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts, batch['ignore_kpt'])

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        # Custom triplet loss for contrastive learning
        """
        # gt_labels (B, n_gt_labels, 1)
        # target_labels (B, n_anchors)
        # embeddings (B, n_features, n_anchors)
        # fg_mask (B, n_anchors) <-- boolean vector showing which anchor (detection candidate) has a match with the GT boxes
        """
        triplet_loss = 0
        logits_avg = 0

        gt_labels = gt_labels.squeeze(-1)  # (B, n_gt_labels)

        for b in range(batch_size):
            if not fg_mask[b].any():
                continue
            # apply foreground masking
            targets = gt_labels[b][target_gt_idx[b].long()[fg_mask[b]]]  # ground truth label for each matched anchor
            embs = embeddings[b][fg_mask[b]]  # embedding for each matched anchor (n_matches, n_features)

            # find unique classes
            uniq_classes = torch.unique(targets)

            # skip if there are not enough classes to make a triplet (it's ok to have query and positive samples from the same class)
            if len(uniq_classes[uniq_classes != 0]) < 2:  # note that class 0 won't be used in the triplet loss
                continue

            # crop species: 1 <= x <= 9
            uniq_crop_classes = torch.tensor(list(filter(lambda x: 1 <= x <= 9, uniq_classes)))  # unique crop classes

            # unknown crop: 10
            unknown_crop_class = 10

            # weed species: 11 <= x <= 19
            uniq_weed_classes = torch.tensor(list(filter(lambda x: 11 <= x <= 19, uniq_classes)))  # unique weed classes

            # unknown weed: 20
            unknown_weed_class = 20

            def select_rand_idx(n_samples, targets, class_label):
                """
                Select n_samples of random indices.
                - 1 random index per row (sample targets)
                - Each index represents a given class.
                Args:
                    n_samples: number of samples
                    targets: target labels that have a match with an anchor candidate (n_matches,)
                    class_label: class label to pick random indices from targets
                Returns:
                    selected_indices: randomly selected indices (n_samples,)

                Example:
                n_samples = 3, class_label = 2
                == n_samples x n_matches ==
                    3 0 1 4 5 2 2 1     <- targets
                    3 0 1 4 5 2 2 1     <- targets
                    3 0 1 4 5 2 2 1     <- targets
                ===========================
                            ...
                    =================
                    0 0 0 0 0 1 1 0  <- masked targets (by class_label)
                    0 0 0 0 0 1 1 0  <- masked targets (by class_label)
                    0 0 0 0 0 1 1 0  <- masked targets (by class_label)
                    =================
                            ...
                    ===================
                    0 1 2 3 4 *5  6  7  <- * indicates the randomly selected index
                    0 1 2 3 4  5 *6  7  <- * indicates the randomly selected index
                    0 1 2 3 4 *5  6  7  <- * indicates the randomly selected index
                    ===================
                            ...
                    selected_indices = [5, 6, 5]
                """
                # upsample class label n times
                class_labels = torch.tensor([class_label] * n_samples, device=targets.device)  # (n_samples,)
                # create a boolean mask where each row shows which target label == class label
                mask = targets.unsqueeze(0) == class_labels.unsqueeze(1)  # (n_samples, n_matches)
                # convert mask to float for torch.multinomial
                probs = mask.float()
                # take one index per row (target labels) where mask is True
                selected_indices = torch.multinomial(probs, num_samples=1).squeeze(1)  # (n_samples,)
                return selected_indices
            
            # make triplets for each case
            """
                crop-i, crop-i, weed-j (common)
                crop-i, crop-i, crop-k (rare)
                crop-i, crop-i, unknown-weed (common for data not labeled with weed species)
                weed-j, weed-j, crop-i (common)
                weed-j, weed-j, weed-k (common)
                weed-j, weed-j, unknown-crop (very rare)
                unknown-weed, itself, crop-i (common for data not labeled with weed species)
                unknown-crop, itself, weed-j (very rare)
            """
            crops_exist = len(uniq_crop_classes) != 0  # do crop classes exist?
            weed_exist = len(uniq_weed_classes) != 0  # do weed classes exist?
            unknown_crop_exist = unknown_crop_class in uniq_classes  # does unknown_crop_class exist?
            unknown_weed_exist = unknown_weed_class in uniq_classes  # does unknown_weed_class exist?

            class_pairs = []
            if crops_exist and weed_exist:
                crop_weed_product = list(itertools.product(uniq_crop_classes, uniq_weed_classes))
                # case1: (crop i, crop i, weed j)
                class_pairs.extend([(crop_class, weed_class) for crop_class, weed_class in crop_weed_product])
                # case4: (weed j, weed j, crop i)
                class_pairs.extend([(weed_class, crop_class) for crop_class, weed_class in crop_weed_product])
            if crops_exist:
                # case2: (crop i, crop i, crop k)
                crop1_crop2_product = list(itertools.product(uniq_crop_classes, repeat=2))
                class_pairs.extend([(crop1_class, crop2_class) for crop1_class, crop2_class in crop1_crop2_product])
            if crops_exist and unknown_weed_exist:
                # case3: (crop i, crop i, unknown weed) & case7: (unknown weed, itself, crop i)
                for crop_class in uniq_crop_classes:
                    class_pairs.extend([(crop_class, unknown_weed_class), (unknown_weed_class, crop_class)])
            if weed_exist:
                # case5: (weed j, weed j, weed k)
                weed1_weed2_product = list(itertools.product(uniq_weed_classes, repeat=2))
                class_pairs.extend([(weed1_class, weed2_class) for weed1_class, weed2_class in weed1_weed2_product])
            if weed_exist and unknown_crop_exist:
                # case6: (weed j, weed j, unknown crop) & case8: (unknown crop, itself, weed j)
                for weed_class in uniq_weed_classes:
                    class_pairs.extend([(weed_class, unknown_crop_class), (unknown_crop_class, weed_class)])

            if not class_pairs:
                continue

            # sample random n pairs
            n_samples = self.n_samples
            pair_samples = [class_pairs[i] for i in torch.randint(0, len(class_pairs), (n_samples,))]  # redundant pairs are allowed

            # (query, positive, negative)
            triplets = []
            for pair in pair_samples:
                class_a, class_b = pair  # (class A, class B)
                a = select_rand_idx(1, targets, class_a)  # (1,)
                b = select_rand_idx(1, targets, class_b)  # (1,)
                if class_a == class_b:
                    triplets.append((a, b, a))  # b should be in front of a
                    triplets.append((b, a, b))  # a should be in front of b
                else:
                    triplets.append((a, a, b))  # b should be behind a
                    triplets.append((b, b, a))  # a should be behind b

            # extract embeddings for query, positive, and negative samples
            query_indices = torch.cat([triplet[0] for triplet in triplets])  # query sample indices (n_samples,)
            pos_indices = torch.cat([triplet[1] for triplet in triplets])  # positive sample indices (n_samples,)
            neg_indices = torch.cat([triplet[2] for triplet in triplets])  # negative sample indices (n_samples,)

            # extract embeddings for query, positive, and negative samples
            embs_query = embs[query_indices]  # query embeddings (n_samples, n_features)
            embs_pos = embs[pos_indices]  # positive embedding (n_samples, n_features)
            embs_neg = embs[neg_indices]  # negative embeddings (n_samples, n_features)

            # compute triplet loss (vectorization method)
            # equivalent to this brute force method:
            # for i in range(n_samples):
            #   triplets = (embs_query[i], embs_pos[i], embs_neg[i])  # embeddings for query, positive, and negative samples
            #   logit = torch.dot(triplets[0], (triplets[1] - triplets[2]))  # dot(query, pos) - dot(query, neg)
            #   y = torch.ones_like(logit)  # (n_samples,)
            #   triplet_loss += F.binary_cross_entropy_with_logits(logit, y)
            diff = embs_pos - embs_neg  # (n_samples, n_features)
            logits = torch.sum(embs_query * diff, dim=1)  # (n_samples,)
            logits_avg += logits.mean().item()
            y = torch.ones_like(logits)  # y is always 1 if we consider (query, pos) as a positive pair; (n_samples,)
            triplet_loss += F.binary_cross_entropy_with_logits(logits, y, reduction='sum') / n_samples  # normalize by number of samples

        # compute triplet loss
        triplet_loss /= batch_size
        loss[5] = triplet_loss

        # compute average logits for monitoring (not for backprop)
        logits_avg /= batch_size
        loss[6] = logits_avg

        return loss[:6].sum() * batch_size, loss.detach()  # loss(box, pose, kobj, cls, dfl, triplet)


class v8PoseMultiClsHeadsLoss(v8PoseLoss):
    """Criterion class for computing training losses for a pose model with multiple classification heads."""

    def __init__(self, model):  # model must be de-paralleled
        """Initializes v8PoseMultiClsHeadsLoss with model, sets keypoint variables and declares a keypoint loss instance."""
        super().__init__(model)
        self.n_losses = 5  # box, cls, dfl, kpt_location, kpt_visibility
        self.nc_per_head = model.model[-1].nc_per_head  # number of classes per classification head

    def __call__(self, preds, batch):
        """Calculate the total loss and detach it."""
        loss = torch.zeros(self.n_losses, device=self.device)  # box, cls, dfl, kpt_location, kpt_visibility
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)

        # B, grids, ..
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # image size (h,w)
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        batch_size = pred_scores.shape[0]
        batch_idx = batch['batch_idx'].view(-1, 1)
        targets = torch.cat((batch_idx, batch['cls'].view(-1, 1), batch['bboxes']), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        # if no targets across the batch, multiply 0 to not affect training while allowing gradient flow in GradScaler backprop
        if gt_labels.shape[1] == 0:
            zero_loss = abs(torch.zeros(self.n_losses, device=self.device)  * pred_scores.sum())
            return zero_loss.sum() * batch_size, zero_loss.detach()
        
        # assign a classification head index to each image in the batch by reasoning on the very first target's class
        head_classes = gt_labels[:, 0].int().squeeze() // self.nc_per_head

        # convert gt_label indices to specific classes for each head
        gt_labels = gt_labels % self.nc_per_head

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))  # (b, h*w, 17, 3)

        # for each batch (B) & all anchors (N), drop all irrelevant classes (C) from classification predictions
        for b, c in enumerate(head_classes):  # each batch represents an image (and a corresponding cls head)
            c_offset = c * self.nc_per_head
            pred_scores[b, :, 0:self.nc_per_head] = pred_scores[b, :, c_offset:c_offset+self.nc_per_head]  # bring classes the front
        pred_scores = pred_scores[:, :, 0:self.nc_per_head]  # slice to keep only the classes of the head

        # target_bboxes: (B, N, 4), target_scores: (B, N, C), fg_mask: (B, N), target_gt_idx: (B, N)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(), (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        # slice target scores as prediction scores are sliced
        target_scores = target_scores[:, :, 0:self.nc_per_head]

        target_scores_sum = max(target_scores.sum(), 1)

        # Cls loss
        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[4] = self.bbox_loss(pred_distri, pred_bboxes, anchor_points, target_bboxes, target_scores,
                                              target_scores_sum, fg_mask)
            keypoints = batch['keypoints'].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(fg_mask, target_gt_idx, keypoints, batch_idx,
                                                             stride_tensor, target_bboxes, pred_kpts, batch['ignore_kpt'])

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.pose  # pose gain
        loss[2] *= self.hyp.kobj  # kobj gain
        loss[3] *= self.hyp.cls  # cls gain
        loss[4] *= self.hyp.dfl  # dfl gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, dfl)


class v8ClassificationLoss:
    """Criterion class for computing training losses."""

    def __call__(self, preds, batch):
        """Compute the classification loss between predictions and true labels."""
        loss = torch.nn.functional.cross_entropy(preds, batch['cls'], reduction='sum') / 64
        loss_items = loss.detach()
        return loss, loss_items
