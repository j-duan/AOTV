import torch
import numpy as np
import nibabel as nib
import cv2, math
import torch.nn.functional as f


# *====================================================================================================================*
# |                                               Miscellaneous                                                        |
# *====================================================================================================================*

def pad_image(image, dim):
    """
     pad image so that the size can be divided by 8 """
    
    if dim == 2:
        X = image.shape[0]
        Y = image.shape[1]
        X2, Y2 = int(math.ceil(X / 8.0)) * 8, int(math.ceil(Y / 8.0)) * 8
        x_pre, y_pre = int((X2 - X) / 2), int((Y2 - Y) / 2)
        x_post, y_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (0, 0), (0, 0)), 'constant')
        
    if dim == 3:
        X = image.shape[0]
        Y = image.shape[1]
        Z = image.shape[2]
        X2, Y2, Z2 = int(math.ceil(X / 8.0)) * 8, int(math.ceil(Y / 8.0)) * 8, int(math.ceil(Z / 8.0)) * 8
        x_pre,  y_pre,  z_pre  = int((X2 - X) / 2), int((Y2 - Y) / 2), int((Z2 - Z) / 2)
        x_post, y_post, z_post = (X2 - X) - x_pre, (Y2 - Y) - y_pre, (Z2 - Z) - z_pre
        image = np.pad(image, ((x_pre, x_post), (y_pre, y_post), (z_pre, z_post)), 'constant')

    return image


def rescale_intensity(image, thresh=(1.0, 99.0)):
    """
     Rescale the image intensity to the range of [0, 1] """
    image = image.astype(np.float32)
    val_l, val_h = np.percentile(image, thresh)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def read_image(path, rescale=True, device='cpu'):
    if path[-7:] == ".nii.gz":
        image = nib.load(path)
        image = image.get_fdata()[:, :, :]

        if rescale:
            image = rescale_intensity(image)

        # image = torch.from_numpy(image).squeeze().unsqueeze(0).unsqueeze(0).to(device)

        return image.squeeze()

    elif path[-4:] == ".png":
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if rescale:
            image = rescale_intensity(image)

        # image = torch.from_numpy(image).squeeze().unsqueeze(0).unsqueeze(0).to(device)

        return image

    else:
        print("Error: uknown image format.")

