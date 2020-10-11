from .statistics import get_mean_std

import cv2
import numpy as np
import os


def visualize_flow(flo, save_path, img_name):
    mean, std = get_mean_std()

    flo = flo.to("cpu")

    # Standardization
    flo[0] = (flo[0] * std[0]) + mean[0]
    flo[1] = (flo[1] * std[1]) + mean[1]

    np_flo = flo.detach().numpy()
    np_flo = np_flo.transpose((1, 2, 0))

    mag, ang = cv2.cartToPolar(np_flo[..., 0], np_flo[..., 1])
    hsv = np.zeros((np_flo.shape[0], np_flo.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(save_path, img_name), bgr_img)
