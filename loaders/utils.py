import sys

import numpy as np


def install_numpy_pickle_compat_aliases():
    # Allow pickles produced by newer NumPy versions to load in older envs.
    sys.modules.setdefault('numpy._core', np.core)
    sys.modules.setdefault('numpy._core.multiarray', np.core.multiarray)
    sys.modules.setdefault('numpy._core.numeric', np.core.numeric)


def compose_ego2img(ego2global_t,
                    ego2global_r,
                    sensor2global_t,
                    sensor2global_r,
                    cam_intrinsic):
    ego2global_t = np.asarray(ego2global_t, dtype=np.float64).reshape(3)
    ego2global_r = np.asarray(ego2global_r, dtype=np.float64).reshape(3, 3)
    sensor2global_t = np.asarray(sensor2global_t, dtype=np.float64).reshape(3)
    sensor2global_r = np.asarray(sensor2global_r, dtype=np.float64).reshape(3, 3)
    cam_intrinsic = np.asarray(cam_intrinsic, dtype=np.float64)

    R = np.linalg.inv(sensor2global_r) @ ego2global_r
    # (ego2global_t - sensor2global_t) @ _inv(sensor2global_r).T
    # = (ego2global_t - sensor2global_t) @ sensor2global_r
    T = (ego2global_t - sensor2global_t) @ sensor2global_r

    ego2cam_rt = np.eye(4)
    ego2cam_rt[:3, :3] = R
    ego2cam_rt[:3, 3] = T.T

    viewpad = np.eye(4)
    viewpad[:cam_intrinsic.shape[0], :cam_intrinsic.shape[1]] = cam_intrinsic
    ego2img = (viewpad @ ego2cam_rt).astype(np.float32)

    return ego2img
