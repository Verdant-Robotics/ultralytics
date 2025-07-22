import numpy as np
import torch


class Shuffler:
    def __init__(self, tile_shape=None, num_oper_range=None, operations=None, scale=1):
        if operations:
            self.operations = operations
        else:
            self.operations = self._create_operations(tile_shape, num_oper_range, scale)

    def _create_operations(self, tile_shape, num_oper_range, scale):
        num_oper_x = np.random.randint(num_oper_range[0], num_oper_range[1])
        num_oper_y = np.random.randint(num_oper_range[0], num_oper_range[1])
        H, W = tile_shape
        operations = []
        for _ in range(num_oper_x):
            y_ran = np.random.randint(0, H // scale) * scale
            delta_x_top_ran = np.random.randint(0, W // scale) * scale
            delta_x_bottom_ran = np.random.randint(0, W // scale) * scale
            smax = SliceAndMoveX(
                y=y_ran, delta_x_top=delta_x_top_ran, delta_x_bottom=delta_x_bottom_ran
            )
            operations.append(smax)
        for _ in range(num_oper_y):
            x_ran = np.random.randint(0, W // scale) * scale
            delta_y_left_ran = np.random.randint(0, H // scale) * scale
            delta_y_right_ran = np.random.randint(0, H // scale) * scale
            smay = SliceAndMoveY(
                x=x_ran, delta_y_left=delta_y_left_ran, delta_y_right=delta_y_right_ran
            )
            operations.append(smay)
        np.random.shuffle(operations)
        return operations

    def scale(self, scale):
        scale_h, scale_w = scale
        scaled_operations = []
        for operation in self.operations:
            scaled_operation = operation.scale(scale_h, scale_w)
            scaled_operations.append(scaled_operation)
        return Shuffler(operations=scaled_operations)

    def shuffle(self, tile):
        for operation in self.operations:
            tile = operation.apply(tile)
        return tile

    def unshuffle(self, tile):
        for operation in reversed(self.operations):
            tile = operation.reverse(tile)
        return tile


class SliceAndMove:
    def __init__(self, split_pos, delta_first, delta_second):
        self.split_pos = split_pos
        self.delta_first = delta_first
        self.delta_second = delta_second
        self.split_dim = None
        self.move_dim = None

    def apply(self, tile_tensor):  # Tile tensor has a shape of H, W, C
        first = self._slice_first(tile_tensor)
        second = self._slice_second(tile_tensor)    
        first = np.roll(first, shift=self.delta_first, axis=self.move_dim)
        second = np.roll(second, shift=self.delta_second, axis=self.move_dim)
        return np.concatenate([first, second], axis=self.split_dim)

    def reverse(self, tile_tensor, scale=None):
        first = self._slice_first(tile_tensor)
        second = self._slice_second(tile_tensor)
        first = np.roll(first, shift=-self.delta_first, axis=self.move_dim)
        second = np.roll(second, shift=-self.delta_second, axis=self.move_dim)
        combined = np.concatenate([first, second], axis=self.split_dim)
        return combined

    def _slice_first(self, tile_tensor):
        if self.split_dim == 1:
            return tile_tensor[:, : self.split_pos, :]
        else:
            return tile_tensor[ : self.split_pos, :,:]

    def _slice_second(self, tile_tensor):
        if self.split_dim == 1:
            return tile_tensor[:, self.split_pos :, :]
        else:
            return tile_tensor[self.split_pos :, :, :]


class SliceAndMoveX(SliceAndMove):
    def __init__(self, y, delta_x_top, delta_x_bottom):
        super().__init__(split_pos=y, delta_first=delta_x_top, delta_second=delta_x_bottom)
        self.split_dim = 0  # Split along Y-axis
        self.move_dim = 1  # Move along X-axis

    def scale(self, scale_h, scale_w):
        return SliceAndMoveX(
            y=int(self.split_pos * scale_h),
            delta_x_top=int(self.delta_first * scale_w),
            delta_x_bottom=int(self.delta_second * scale_w),
        )


class SliceAndMoveY(SliceAndMove):
    def __init__(self, x, delta_y_left, delta_y_right):
        super().__init__(split_pos=x, delta_first=delta_y_left, delta_second=delta_y_right)
        self.split_dim = 1  # Split along X-axis
        self.move_dim = 0  # Move along Y-axis

    def scale(self, scale_h, scale_w):
        return SliceAndMoveY(
            x=int(self.split_pos * scale_w),
            delta_y_left=int(self.delta_first * scale_h),
            delta_y_right=int(self.delta_second * scale_h),
        )


class MoveXY:
    def __init__(self, delta_x, delta_y):
        self.delta_x = delta_x
        self.delta_y = delta_y

    def apply(self, tile_tensor):
        tile_tensor = torch.roll(tile_tensor, shifts=self.delta_x, dims=2)
        tile_tensor = torch.roll(tile_tensor, shifts=self.delta_y, dims=1)
        return tile_tensor

    def reverse(self, tile_tensor):
        tile_tensor = torch.roll(tile_tensor, shifts=-self.delta_y, dims=1)
        tile_tensor = torch.roll(tile_tensor, shifts=-self.delta_x, dims=2)
        return tile_tensor
