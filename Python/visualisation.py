import cv2
import numpy as np


def jac_det_3D(disp):
    """
    Compute Jacobian determinant using displacements
    Args: input should be numpy arrary
        disp: size 3xHxWxD
        disp[0]: vertical displacement
        disp[1]: horizontal displacement
        disp[2]: depth displacement

    Returns: Jacobian determinant HxWxD
    """
    
    # displacement fields in pixel unit
    disp_x, disp_y, disp_z = disp[0], disp[1], disp[2]

    disp_xx, disp_xy, disp_xz = compute_derivatives_3D(disp_x)
    disp_yx, disp_yy, disp_yz = compute_derivatives_3D(disp_y)
    disp_zx, disp_zy, disp_zz = compute_derivatives_3D(disp_z)

    # Add in jac det of ID which is equivalent to adding 1 to diagonals of matrix.
    disp_xx = disp_xx + 1
    disp_yy = disp_yy + 1
    disp_zz = disp_zz + 1
    
    a, b, c = disp_xx, disp_xy, disp_xz
    d, e, f = disp_yx, disp_yy, disp_yz
    g, h, i = disp_zx, disp_zy, disp_zz
    
    jac_det = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

    return jac_det


def jac_det_2D(disp):
    """
    Compute Jacobian determinant using displacements
    Args: input should be numpy arrary
        disp: size 2xHxW
        disp[0]: vertical displacement
        disp[1]: horizontal displacement

    Returns: Jacobian determinant HxW
    """
    
    # displacement fields in pixel unit
    disp_x, disp_y = disp[0], disp[1]

    disp_xx, disp_xy = compute_derivatives_2D(disp_x)
    disp_yx, disp_yy = compute_derivatives_2D(disp_y)

    # Add in jac det of ID which is equivalent to adding 1 to diagonals of matrix.
    disp_xx = disp_xx + 1
    disp_yy = disp_yy + 1
    
    a, b = disp_xx, disp_xy
    c, d = disp_yx, disp_yy

    jac_det = a*d - b*c
    
    return jac_det


def deform_to_hsv(disp, white_bg=True):
    """
    Args: input should be numpy arrary
        disp: size 2xHxW
        disp[0]: vertical displacement
        disp[1]: horizontal displacement

    Returns:
        hsv_flow: HSV encoded flow converted to RGB (for visualisation), size HxW

    """

    # normalise displacement field
    disp_x, disp_y = disp[0], disp[1]
            
    # convert to polar coordinates
    mag, ang = cv2.cartToPolar(disp_x, disp_y)
    max_mag = np.max(mag)

    # hsv encoding

    # Create a 3 channel array of zeroes
    hsv_flow = np.zeros((disp_x.shape[0], disp_x.shape[1], 3))
    hsv_flow[..., 0] = ang * 180 / np.pi / 2  # hue = angle
    hsv_flow[..., 1] = 255.0  # saturation = 255
    hsv_flow[..., 2] = 255.0 * mag / max_mag

    # convert hsv encoding to rgb for visualisation
    # ([..., ::-1] converts from BGR to RGB)
    hsv_flow = cv2.cvtColor(hsv_flow.astype(np.uint8), cv2.COLOR_HSV2BGR)[..., ::-1]
    hsv_flow = hsv_flow.astype(np.uint8)

    if white_bg:
        hsv_flow = 255 - hsv_flow

    return hsv_flow


def compute_derivatives_2D(img):
    """ 
    input: should be numpy arrary
    arg: img of size (H, W)
    
    return:
    img_x: derivative along vertical direction
    img_y: derivative along horizontal direction
    
    """
    H, W = img.shape[-2:]
    
    C3 = np.roll(img, -1, -2)
    C3[..., H-1, :] = C3[..., H-2, :]
    C4 = np.roll(img,  1, -2)
    C4[..., 0, :] = C4[..., 1, :]
    img_x = (C3 - C4) / 2
    
    C1 = np.roll(img, -1, -1)
    C1[..., W-1] = C1[..., W-2]    
    C2 = np.roll(img,  1, -1)
    C2[..., 0] = C2[..., 1]
    img_y = (C1 - C2) / 2
    
    return img_x, img_y


def compute_derivatives_3D(image):
    """ 
    input: should be numpy arrary
    arg: img of size (H, W, D)
    
    return:
    img_x: derivative along vertical direction
    img_y: derivative along horizontal direction
    img_z: derivative along depth direction
    
    """
    
    H, W, D = image.shape[-3:]
    
    # Finite differences in x dimension
    x1 = np.roll(image, -1, -3)
    x1[..., H - 1, :, :] = x1[..., H - 2, :, :]
    x2 = np.roll(image, 1, -3)
    x2[..., 0, :, :] = x2[..., 1, :, :]
    img_x = (x1 - x2) / 2

    # Finite differences in y dimension
    y1 = np.roll(image, -1, -2)
    y1[..., W - 1, :] = y1[..., W - 2, :]
    y2 = np.roll(image, 1, -2)
    y2[..., 0, :] = y2[..., 1, :]
    img_y = (y1 - y2) / 2
    
    # Finite differences in z dimension
    z1 = np.roll(image, -1, -1)
    z1[..., D - 1] = z1[..., D - 2]
    z2 = np.roll(image, 1, -1)
    z2[..., 0] = z2[..., 1]
    img_z = (z1 - z2) / 2

    return img_x, img_y, img_z