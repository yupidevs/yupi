import json
import os.path
import numpy as np
import matplotlib.pyplot as plt
from yupi.tracking.affine_estimator import affine_matrix


def affine2camera(theta, tx, ty):
    x_cl, y_cl, theta_cl = np.zeros((3, theta.size + 1))
    theta_cl[1:] = np.cumsum(theta)

    for i in range(theta.size):
        A = affine_matrix(theta_cl[i + 1], x_cl[i], y_cl[i], R_inv=True)
        x_cl[i + 1], y_cl[i + 1] = A @ [-tx[i], -ty[i], 1]

    x_cl, y_cl, theta_cl = x_cl[1:], y_cl[1:], theta_cl[1:]
    return x_cl, y_cl, theta_cl


def camera2ant(x_ac, y_ac, x_cl, y_cl, theta_cl):
    x_al, y_al = np.empty((2, x_ac.size))

    for i in range(x_ac.size):
        A = affine_matrix(theta_cl[i], x_cl[i], y_cl[i], R_inv=True)
        x_al[i], y_al[i] = A @ [x_ac[i], y_ac[i], 1]

    return x_al, y_al


def affine2ant(theta, tx, ty, x_ac, y_ac):
    x_cl, y_cl, theta_cl = affine2camera(theta, tx, ty)
    x_al, y_al = camera2ant(x_ac, y_ac, x_cl, y_cl, theta_cl)
    return x_al, y_al



def plot_results(file_dir):
    pix_per_cm = 63.
    pix_per_m = pix_per_cm * 100

    # file_name = 'data_video2_[1.2min-100.0%].json'
    # file_dir = os.path.join('data', file_name)
    with open(file_dir, 'r') as file:
        data = json.load(file)

    # get affine parameters and ant coordinates in the camera frame of reference
    affine_params = np.array(data['affine_params'])
    r_ac = np.array(data['r_ac'], dtype=np.float)

    # unpack and convert from pixels to meters
    theta, tx, ty, _ = affine_params.T
    tx, ty = tx / pix_per_m, ty / pix_per_m
    x_ac, y_ac = r_ac.T / pix_per_m

    # invert axis
    theta *= -1
    ty *= -1
    y_ac *= -1

    # ant position with respect to the lab
    x_al, y_al = affine2ant(theta, tx, ty, x_ac, y_ac)
    x = x_al - x_al[0]
    y = y_al - y_al[0]

