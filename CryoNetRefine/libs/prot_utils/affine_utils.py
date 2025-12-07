"""
MIT License

Copyright (c) 2022 Kiarash Jamali

This file is modified from [https://github.com/3dem/model-angelo/blob/main/model_angelo/utils/affine_utils.py].

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions.
"""
import torch
from .torch_utlis import is_ndarray,shared_cat
from torch import nn
def get_affine(rot_matrix, shift):
    is_torch = torch.is_tensor(rot_matrix) and torch.is_tensor(shift)
    is_numpy = is_ndarray(rot_matrix) and is_ndarray(shift)

    if is_torch or is_numpy:
        if len(rot_matrix.shape) == len(shift.shape):
            return shared_cat((rot_matrix, shift), dim=-1, is_torch=is_torch)
        elif len(rot_matrix.shape) == len(shift.shape) + 1:
            return shared_cat((rot_matrix, shift[..., None]), dim=-1, is_torch=is_torch)
        else:
            raise ValueError(
                f"get_affine does not support rotation matrix of shape {rot_matrix.shape}"
                f"and shift of shape {shift.shape} "
            )
    else:
        raise ValueError(
            f"get_affine does not support different types for rot_matrix and shift, ie one is a numpy array, "
            f"the other is a torch tensor "
        )


def get_affine_translation(affine):
    return affine[..., :, -1]
def get_affine_rot(affine):
    return affine[..., :3, :3]
def invert_affine(affine):
    inv_rots = get_affine_rot(affine).transpose(-1, -2)
    t = torch.einsum("...ij,...j->...i", inv_rots, affine[..., :, -1])
    inv_shift = -t
    return get_affine(inv_rots, inv_shift)
def affine_mul_vecs(affine, vecs):
    num_unsqueeze_dims = len(vecs.shape) - len(affine.shape) + 1
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    return torch.einsum(
        "...ij, ...j-> ...i", get_affine_rot(affine), vecs
    ) + get_affine_translation(affine)
def vecs_to_local_affine(affine, vecs):
    return affine_mul_vecs(invert_affine(affine), vecs)

def grid_sampler_normalize(coord, size, align_corners=False):
    if align_corners:
        return (2 / (size - 1)) * coord - 1
    else:
        return ((2 * coord + 1) / size) - 1


def sample_centered_cube(
    grid,
    rotation_matrices,
    shifts,
    cube_side=10,
    align_corners=True,
):
    assert len(grid.shape) == 5
    bz, cz, szz, szy,szx = grid.shape
    sz = torch.tensor([szx,szy,szz])
    align_d = 1 if align_corners else 0
    scale_mult = (
        (
            torch.Tensor(
                [
                    cube_side,
                    cube_side,
                    cube_side,
                ]
            )
            - align_d
        )
        / (sz - align_d)
    ).to(grid.device)
    center_shift_vector = -torch.Tensor(
        [[cube_side // 2, cube_side // 2, cube_side // 2]]
    ).to(shifts.device)
    center_shift_vector = torch.einsum(
        "...ij, ...j-> ...i", rotation_matrices, center_shift_vector.expand(len(rotation_matrices),3)
    )

    rotation_matrices = rotation_matrices * scale_mult[None][None]
    shifts = (
        grid_sampler_normalize(
            shifts + center_shift_vector, sz.to(shifts.device), align_corners=align_corners
        )
        + rotation_matrices.sum(-1)
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = nn.functional.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + 3 * (cube_side,),
        align_corners=align_corners,
    )

    return nn.functional.grid_sample(grid.detach(), cc, align_corners=align_corners)

def get_z_to_w_rotation_matrix(w):
    """
    Special case of get_a_to_b_rotation matrix for when you are converting from
    the Z axis to some vector w. Algorithm comes from
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    _w = nn.functional.normalize(w, p=2, dim=-1)
    # (1, 0, 0) cross _w
    v2 = -_w[..., 2]
    v3 = _w[..., 1]
    # (1, 0, 0) dot _w
    c = _w[..., 0]
    # The result of I + v_x + v_x_2 / (1 + c)
    # [   1 - (v_2^2 + v_3 ^ 2) / (1 + c),                   -v_3,                    v_2]
    # [                               v_3,    1 - v_3^2 / (1 + c),    v_2 * v_3 / (1 + c)]
    # [                              -v_2,    v_2 * v_3 / (1 + c),    1 - v_2^2 / (1 + c)]
    R = torch.zeros(*w.shape[:-1], 3, 3).to(w.device)
    v2_2, v3_2 = ((v2 ** 2) / (1 + c)), ((v3 ** 2) / (1 + c))
    v2_v3 = v2 * v3 / (1 + c)
    R[..., 0, 0] = 1 - (v2_2 + v3_2)
    R[..., 0, 1] = -v3
    R[..., 0, 2] = v2
    R[..., 1, 0] = v3
    R[..., 1, 1] = 1 - v3_2
    R[..., 1, 2] = v2_v3
    R[..., 2, 0] = -v2
    R[..., 2, 1] = v2_v3
    R[..., 2, 2] = 1 - v2_2
    return R
def sample_centered_rectangle(
    grid,
    rotation_matrices,
    shifts,
    rectangle_length=10,
    rectangle_width=3,
    align_corners=True,
):
    assert len(grid.shape) == 5
    bz, cz, szz, szy, szx = grid.shape
    sz = torch.tensor([szx, szy, szz])
    align_d = 1 if align_corners else 0
    scale_mult = (
        (
            torch.Tensor(
                [
                    rectangle_width,
                    rectangle_width,
                    rectangle_length,
                ]
            )
            - align_d
        )
        / (sz - align_d)
    ).to(grid.device)
    center_shift_vector = -torch.Tensor(
        [[rectangle_width // 2, rectangle_width // 2, rectangle_width // 2]]
    ).to(shifts.device)
    center_shift_vector = torch.einsum(
        "...ij, ...j-> ...i", rotation_matrices, center_shift_vector.expand(len(rotation_matrices),3)
    )
    rotation_matrices = rotation_matrices * scale_mult[None][None]
    shifts = (
        grid_sampler_normalize(
            shifts + center_shift_vector, sz.to(shifts.device), align_corners=align_corners
        )
        + rotation_matrices.sum(-1)
    )
    affine_matrix = get_affine(rotation_matrices, shifts)
    cc = nn.functional.affine_grid(
        affine_matrix,
        (
            bz,
            cz,
        )
        + (rectangle_length, rectangle_width, rectangle_width),
        align_corners=align_corners,
    )
    return nn.functional.grid_sample(grid.detach(), cc, align_corners=align_corners)


def sample_centered_rectangle_along_vector(
    batch_grids,
    batch_vectors,
    batch_origin_points,
    rectangle_length=10,
    rectangle_width=3,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_vectors = [batch_vectors]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, vectors, origin_points) in zip(
        batch_grids, batch_vectors, batch_origin_points
    ):
        rotation_matrices = get_z_to_w_rotation_matrix(vectors)
        rectangle = sample_centered_rectangle(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            rectangle_length=rectangle_length,
            rectangle_width=rectangle_width,
        )
        if marginalization_dims is not None:
            rectangle = rectangle.sum(dim=marginalization_dims)
        output.append(rectangle)
    output = torch.cat(output, dim=0)
    return output
def sample_centered_cube_rot_matrix(
    batch_grids,
    batch_rot_matrices,
    batch_origin_points,
    cube_side=10,
    marginalization_dims=None,
):
    if not isinstance(batch_grids, list):
        batch_grids = [batch_grids]
        batch_rot_matrices = [batch_rot_matrices]
        batch_origin_points = [batch_origin_points]
    output = []
    for (grid, rotation_matrices, origin_points) in zip(
        batch_grids, batch_rot_matrices, batch_origin_points
    ):
        cube = sample_centered_cube(
            grid,
            rotation_matrices.to(grid.device),
            origin_points.to(grid.device),
            cube_side=cube_side,
        )
        if marginalization_dims is not None:
            cube = cube.sum(dim=marginalization_dims)
        output.append(cube)
    output = torch.cat(output, dim=0)
    return output
def rots_from_two_vecs(e1_unnormalized, e2_unnormalized):
    e1 = nn.functional.normalize(e1_unnormalized, p=2, dim=-1)
    c = torch.einsum("...i,...i->...", e2_unnormalized, e1)[..., None]  # dot product
    e2 = e2_unnormalized - c * e1
    e2 = nn.functional.normalize(e2, p=2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack((e1, e2, e3), dim=-1)
def init_random_affine_from_translation(translation):
    v, w = torch.rand_like(translation), torch.rand_like(translation)
    rot = rots_from_two_vecs(v, w)
    return get_affine(rot, translation)
def affine_from_3_points(point_on_neg_x_axis, origin, point_on_xy_plane):
    rotation = rots_from_two_vecs(
        e1_unnormalized=origin - point_on_neg_x_axis,
        e2_unnormalized=point_on_xy_plane - origin,
    )
    return get_affine(rotation, origin)
def affine_from_tensor4x4(m):
    assert m.shape[-1] == 4 == m.shape[-2]
    return get_affine(m[..., :3, :3], m[..., :3, -1])
def fill_rotation_matrix(xx, xy, xz, yx, yy, yz, zx, zy, zz):
    R = torch.zeros(*xx.shape, 3, 3).to(xx.device)
    R[..., 0, 0] = xx
    R[..., 0, 1] = xy
    R[..., 0, 2] = xz

    R[..., 1, 0] = yx
    R[..., 1, 1] = yy
    R[..., 1, 2] = yz

    R[..., 2, 0] = zx
    R[..., 2, 1] = zy
    R[..., 2, 2] = zz
    return R
def affine_mul_rots(affine, rots):
    num_unsqueeze_dims = len(rots.shape) - len(affine.shape)
    if num_unsqueeze_dims > 0:
        new_shape = affine.shape[:-2] + num_unsqueeze_dims * (1,) + (3, 4)
        affine = affine.view(*new_shape)
    rotation = affine[..., :3, :3] @ rots
    return get_affine(rotation, get_affine_translation(affine))
def affine_composition(a1, a2):
    """
    Does the operation a1 o a2
    """
    rotation = get_affine_rot(a1) @ get_affine_rot(a2)
    translation = affine_mul_vecs(a1, get_affine_translation(a2))
    return get_affine(rotation, translation)



def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def rot_matmul(a, b):
    row_1 = torch.stack(
        [
            a[..., 0, 0] * b[..., 0, 0]
            + a[..., 0, 1] * b[..., 1, 0]
            + a[..., 0, 2] * b[..., 2, 0],
            a[..., 0, 0] * b[..., 0, 1]
            + a[..., 0, 1] * b[..., 1, 1]
            + a[..., 0, 2] * b[..., 2, 1],
            a[..., 0, 0] * b[..., 0, 2]
            + a[..., 0, 1] * b[..., 1, 2]
            + a[..., 0, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_2 = torch.stack(
        [
            a[..., 1, 0] * b[..., 0, 0]
            + a[..., 1, 1] * b[..., 1, 0]
            + a[..., 1, 2] * b[..., 2, 0],
            a[..., 1, 0] * b[..., 0, 1]
            + a[..., 1, 1] * b[..., 1, 1]
            + a[..., 1, 2] * b[..., 2, 1],
            a[..., 1, 0] * b[..., 0, 2]
            + a[..., 1, 1] * b[..., 1, 2]
            + a[..., 1, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )
    row_3 = torch.stack(
        [
            a[..., 2, 0] * b[..., 0, 0]
            + a[..., 2, 1] * b[..., 1, 0]
            + a[..., 2, 2] * b[..., 2, 0],
            a[..., 2, 0] * b[..., 0, 1]
            + a[..., 2, 1] * b[..., 1, 1]
            + a[..., 2, 2] * b[..., 2, 1],
            a[..., 2, 0] * b[..., 0, 2]
            + a[..., 2, 1] * b[..., 1, 2]
            + a[..., 2, 2] * b[..., 2, 2],
        ],
        dim=-1,
    )

    return torch.stack([row_1, row_2, row_3], dim=-2)


def rot_vec_mul(r, t):
    x = t[..., 0]
    y = t[..., 1]
    z = t[..., 2]
    return torch.stack(
        [
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )

def convert_3x4_to_4x4_torch(tensor):
    """
    Convert tensor of shape [*, 3, 4] to shape [*, 4, 4].
    :param tensor: Input tensor of shape [*, 3, 4]
    :return: Converted tensor of shape [*, 4, 4]
    """
    # Get the leading dimensions of the input tensor
    leading_dims = tensor.shape[:-2]
    
    # Create a zero tensor of shape [*, 1, 4]
    new_row = torch.zeros(*leading_dims, 1, 4, dtype=tensor.dtype, device=tensor.device)
    # Set the last column to 1
    new_row[..., 0, -1] = 1
    
    # Concatenate along the second-to-last dimension (row direction)
    result = torch.cat((tensor, new_row), dim=-2)
    
    return result


class T:
    def __init__(self, rots, trans):
        self.rots = rots
        self.trans = trans

        if self.rots is None and self.trans is None:
            raise ValueError("Only one of rots and trans can be None")
        elif self.rots is None:
            self.rots = T.identity_rot(
                self.trans.shape[:-1],
                self.trans.dtype,
                self.trans.device,
                self.trans.requires_grad,
            )
        elif self.trans is None:
            self.trans = T.identity_trans(
                self.rots.shape[:-2],
                self.rots.dtype,
                self.rots.device,
                self.rots.requires_grad,
            )

        if (
            self.rots.shape[-2:] != (3, 3)
            or self.trans.shape[-1] != 3
            or self.rots.shape[:-2] != self.trans.shape[:-1]
        ):
            raise ValueError("Incorrectly shaped input")

    def __getitem__(self, index):
        if type(index) != tuple:
            index = (index,)
        return T(
            self.rots[index + (slice(None), slice(None))],
            self.trans[index + (slice(None),)],
        )

    def __eq__(self, obj):
        return torch.all(self.rots == obj.rots) and torch.all(
            self.trans == obj.trans
        )

    def __mul__(self, right):
        rots = self.rots * right[..., None, None]
        trans = self.trans * right[..., None]

        return T(rots, trans)

    def __rmul__(self, left):
        return self.__mul__(left)

    @property
    def shape(self):
        s = self.rots.shape[:-2]
        return s if len(s) > 0 else torch.Size([1])

    def get_trans(self):
        return self.trans

    def get_rots(self):
        return self.rots

    def compose(self, t):
        rot_1, trn_1 = self.rots, self.trans
        rot_2, trn_2 = t.rots, t.trans

        rot = rot_matmul(rot_1, rot_2)
        trn = rot_vec_mul(rot_1, trn_2) + trn_1

        return T(rot, trn)

    def apply(self, pts):
        r, t = self.rots, self.trans
        rotated = rot_vec_mul(r, pts)
        return rotated + t

    def invert_apply(self, pts):
        r, t = self.rots, self.trans
        pts = pts - t
        return rot_vec_mul(r.transpose(-1, -2), pts)

    def invert(self):
        rot_inv = self.rots.transpose(-1, -2)
        trn_inv = rot_vec_mul(rot_inv, self.trans)

        return T(rot_inv, -1 * trn_inv)

    def unsqueeze(self, dim):
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self.rots.unsqueeze(dim if dim >= 0 else dim - 2)
        trans = self.trans.unsqueeze(dim if dim >= 0 else dim - 1)

        return T(rots, trans)

    @staticmethod
    def identity_rot(shape, dtype, device, requires_grad):
        rots = torch.eye(
            3, dtype=dtype, device=device, requires_grad=requires_grad
        )
        rots = rots.view(*((1,) * len(shape)), 3, 3)
        rots = rots.expand(*shape, -1, -1)

        return rots

    @staticmethod
    def identity_trans(shape, dtype, device, requires_grad):
        trans = torch.zeros(
            (*shape, 3), dtype=dtype, device=device, requires_grad=requires_grad
        )
        return trans

    @staticmethod
    def identity(shape, dtype, device, requires_grad=True):
        return T(
            T.identity_rot(shape, dtype, device, requires_grad),
            T.identity_trans(shape, dtype, device, requires_grad),
        )

    @staticmethod
    def from_4x4(t):
        rots = t[..., :3, :3]
        trans = t[..., :3, 3]
        return T(rots, trans)

    def to_4x4(self):
        tensor = self.rots.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self.rots
        tensor[..., :3, 3] = self.trans
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor(t):
        return T.from_4x4(t)

    @staticmethod
    def from_3_points(p_neg_x_axis, origin, p_xy_plane, eps=1e-8):
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return T(rots, torch.stack(origin, dim=-1))

    @staticmethod
    def concat(ts, dim):
        rots = torch.cat([t.rots for t in ts], dim=dim if dim >= 0 else dim - 2)
        trans = torch.cat(
            [t.trans for t in ts], dim=dim if dim >= 0 else dim - 1
        )

        return T(rots, trans)

    def map_tensor_fn(self, fn):
        """
        Apply a function that takes a tensor as its only argument to the
        rotations and translations, treating the final two/one
        dimension(s), respectively, as batch dimensions.

        E.g.: Given t, an instance of T of shape [N, M], this function can
        be used to sum out the second dimension thereof as follows:

            t = t.map_tensor_fn(lambda x: torch.sum(x, dim=-1))

        The resulting object has rotations of shape [N, 3, 3] and
        translations of shape [N, 3]
        """
        rots = self.rots.view(*self.rots.shape[:-2], 9)
        rots = torch.stack(list(map(fn, torch.unbind(rots, -1))), dim=-1)
        rots = rots.view(*rots.shape[:-1], 3, 3)

        trans = torch.stack(list(map(fn, torch.unbind(self.trans, -1))), dim=-1)

        return T(rots, trans)

    def stop_rot_gradient(self):
        return T(self.rots.detach(), self.trans)

    def scale_translation(self, factor):
        return T(self.rots, self.trans * factor)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        translation = -1 * c_xyz
        n_xyz = n_xyz + translation
        c_xyz = c_xyz + translation

        c_x, c_y, c_z = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm
        zeros = sin_c1.new_zeros(sin_c1.shape)
        ones = sin_c1.new_ones(sin_c1.shape)

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x ** 2 + c_y ** 2 + c_z ** 2)
        sin_c2 = c_z / norm
        cos_c2 = torch.sqrt(c_x ** 2 + c_y ** 2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c1_rots[..., 2, 0] = -1 * sin_c2
        c1_rots[..., 2, 2] = cos_c2

        c_rots = rot_matmul(c2_rots, c1_rots)
        n_xyz = rot_vec_mul(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y ** 2 + n_z ** 2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = rot_matmul(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        translation = -1 * translation

        return T(rots, translation)

    def cuda(self):
        return T(self.rots.cuda(), self.trans.cuda())
