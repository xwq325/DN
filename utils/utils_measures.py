import numpy as np
import cv2
import math
import phasepack.phasecong as pc
import pyiqa
import torch


# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr_path(path1, path2, test_y_channel=True, color_space="Ycbcr"):
    device = torch.device("cpu")
    iqa_metric1 = pyiqa.create_metric('psnr', test_y_channel=test_y_channel, color_space=color_space, device=device)
    return iqa_metric1(path1, path2).item()

# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim_path(path1, path2, test_y_channel=True, color_space="Ycbcr"):
    device = torch.device("cpu")
    iqa_metric2 = pyiqa.create_metric('ssim', test_y_channel=test_y_channel, color_space=color_space, device=device)
    return iqa_metric2(path1, path2).item()

# --------------------------------------------
# FSIM
# --------------------------------------------
def calculate_fsim_path(path1, path2, chromatic=False):
    device = torch.device("cpu")
    iqa_metric3 = pyiqa.create_metric('fsim', chromatic=chromatic, device=device)
    return iqa_metric3(path1, path2).item()
    

# --------------------------------------------
# LPIPS
# --------------------------------------------
def calculate_lpips_path(path1, path2, net='alex'):
    device = torch.device("cpu")
    iqa_metric4 = pyiqa.create_metric('lpips', net=net, device=device)
    return iqa_metric4(path1, path2).item()
    
# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, test_y_channel=True, color_space="Ycbcr"):
    device = torch.device("cpu")
    iqa_metric1 = pyiqa.create_metric('psnr', test_y_channel=test_y_channel, color_space=color_space, device=device)
    return iqa_metric1(img1, img2).item()

# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, test_y_channel=True, color_space="Ycbcr"):
    device = torch.device("cpu")
    iqa_metric2 = pyiqa.create_metric('ssim', test_y_channel=test_y_channel, color_space=color_space, device=device)
    return iqa_metric2(img1, img2).item()

# --------------------------------------------
# FSIM
# --------------------------------------------
def calculate_fsim(img1, img2, chromatic=False):
    device = torch.device("cpu")
    iqa_metric3 = pyiqa.create_metric('fsim', chromatic=chromatic, device=device)
    return iqa_metric3(img1, img2).item()
    

# --------------------------------------------
# LPIPS
# --------------------------------------------
def calculate_lpips(img1, img2, net='alex'):
    device = torch.device("cpu")
    iqa_metric4 = pyiqa.create_metric('lpips', net=net, device=device)
    return iqa_metric4(img1, img2).item()