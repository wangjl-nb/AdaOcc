import numpy as np


def quaternion_to_matrix(quaternion):
    """Convert a quaternion in (w, x, y, z) order to a rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f'Quaternion must have 4 elements, got {q.shape[0]}')

    norm = np.linalg.norm(q)
    if norm <= 0:
        raise ValueError('Zero-norm quaternion is not valid')
    w, x, y, z = q / norm

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def as_rotation_matrix(rotation):
    """Return a 3x3 rotation matrix from either a matrix or quaternion."""
    rot = np.asarray(rotation)
    if rot.shape == (3, 3):
        return rot.astype(np.float64)
    if rot.size == 4:
        return quaternion_to_matrix(rot)
    raise ValueError(f'Unsupported rotation shape: {rot.shape}')


def transform_matrix(translation, rotation, inverse=False):
    """Build a 4x4 rigid transform matrix.

    Args:
        translation (array-like): Translation vector (x, y, z).
        rotation (array-like): Quaternion (w, x, y, z) or 3x3 rotation matrix.
        inverse (bool): Whether to return the inverse transform.
    """
    trans = np.asarray(translation, dtype=np.float64).reshape(-1)
    if trans.shape[0] != 3:
        raise ValueError(f'Translation must have 3 elements, got {trans.shape[0]}')

    rot_mat = as_rotation_matrix(rotation)

    matrix = np.eye(4, dtype=np.float64)
    if inverse:
        rot_inv = rot_mat.T
        matrix[:3, :3] = rot_inv
        matrix[:3, 3] = -rot_inv @ trans
    else:
        matrix[:3, :3] = rot_mat
        matrix[:3, 3] = trans
    return matrix
