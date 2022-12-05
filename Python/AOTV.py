import torch
import torch.nn.functional as f
from TVL1 import tvl1_admm_optimizer
from SOTVL1 import sotvl1_admm_optimizer
from TOTVL1 import totvl1_admm_optimizer
from FOTVL1 import fotvl1_admm_optimizer

# from IPython.display import clear_output
def AOTV(source, target, levels=[4,2,1], lmbda=0.1, 
         taylor=5, max_iter=500, tol1=1e-3, 
         tol2=1e-2, alpha=1.8, acc=True, align=True, 
         method='1stO-TV', solver='fft', device='cpu'):
   
    """
    Arbitrary Order Total Variation

    Args:
        source: source image, Tensor of shape (N, C, H, W)
        target: target image, Tensor of shape (N, C, H, W)
        level: multi scale optimisation
        lmbda: regularisation parameter
        taylor: numbers of taylor expansion
        max_iter: numbers of over-relaxed ADMM
        tol1: tolerance to break ADMM loop
        tol2: tolerance to break taylor expansion loop
        alpha: over-relexed parameter
        acc: whether to accelerate ADMM via over-relaxation
        align: align corners
        method: '1stO-TV', '2ndO-TV', '3rdO-TV', '4thO-TV'
        solver: 'fft' and 'dct' depending on boundary conditions
        device: cpu or cuda
    Returns:
        displacemnt: shape (N, 2, H, W)
            displacemnt[:,0]: vertical change in pixel unit (N, H, W) 
            displacemnt[:,1]: horizontal change in pixel unit (N, H, W)

    """
    
    stop = []
    for i in range(0,len(levels)):
        # print('\n* Registration at scale {} running on {} using {} method \n'.format(i+1, device, method))

        # Resizing images
        source_ = f.interpolate(source, scale_factor=1/levels[i], mode='bilinear', align_corners=align)
        target_ = f.interpolate(target, scale_factor=1/levels[i], mode='bilinear', align_corners=align)
            
        if levels[i] == levels[0]:
            # Initialise flows as ID grid if first iteration of first level
            N, _, H, W = source_.shape
            disp = torch.zeros(N, 2, H, W, device=torch.device(device)) # N2HW
        else:
            # Resize previously calculated flow for proceeding levels
            disp = f.interpolate(disp, scale_factor=2, mode='bilinear', align_corners=align)*2 # N2HW
            
        for j in range(1, taylor):
            # print('\n*  Taylor expansion {} \n'.format(j))
            ###################################################################
            warped_source = warp_via_displacement(source_, disp, device=device) # p is N2HW
            if method == '1stO-TV':
                disp =  tvl1_admm_optimizer(warped_source, target_, disp, lmbda=lmbda, 
                                           tol=tol1, maxIter=max_iter, alpha=alpha, acc=acc,
                                           solver=solver, device=device) # N2HW
            elif method == '2ndO-TV':
                disp = sotvl1_admm_optimizer(warped_source, target_, disp, lmbda=lmbda, 
                                           tol=tol1, maxIter=max_iter, alpha=alpha, acc=acc,
                                           solver=solver, device=device) # N2HW
            elif method == '3rdO-TV':
                disp = totvl1_admm_optimizer(warped_source, target_, disp, lmbda=lmbda, 
                                           tol=tol1, maxIter=max_iter, alpha=alpha, acc=acc,
                                           solver=solver, device=device) # N2HW 
            else: 
                disp = fotvl1_admm_optimizer(warped_source, target_, disp, lmbda=lmbda, 
                                           tol=tol1, maxIter=max_iter, alpha=alpha, acc=acc,
                                           solver=solver, device=device) # N2HW
                 
            ###################################################################           
            diff = torch.mean((warped_source-target_)**2)
            stop.append(diff)
            if j > 1:
                if abs(stop[-2]-stop[-1])/stop[-1] < tol2:
                    # print('  Taylor expansion converges to tolerance {} \n'.format(tol2))
                    break # set tol to break iterion
            
    return disp


def warp_via_displacement(source, displacement, interp='bilinear', padding='reflection', align=True, device='cpu'):
    """
    warp a 2D image with displacement

    Args:
        source: source image, Tensor of shape (N, C, H, W)
        displacemnt: shape (N, 2, H, W)
            displacemnt[:,0]: vertical change in pixel unit (N, H, W) 
            displacemnt[:,1]: horizontal change in pixel unit (N, H, W)
        interp: method of interpolation
        padding: method of padding boundaries
        align: align corners
    Returns:
        source image deformed using the deformations (N, 1, H, W)

    """
    
    # generate identity mesh grid
    # id_h is vertical grid and id_w is horizontal grid
    H, W = source.size()[-2:]
    id_h, id_w = torch.meshgrid([torch.linspace(-1, 1, H, device=torch.device(device)), 
                                 torch.linspace(-1, 1, W, device=torch.device(device))])
    
    # (H, W) + (N, H, W) add by broadcasting
    deform_h = id_h + displacement[:,0] * 2 / H # normaised vertical displacement of size NHW
    deform_w = id_w + displacement[:,1] * 2 / W # normaised horizontal displacement of size NHW
    deformation = torch.stack((deform_w, deform_h), -1)  # shape (N, H, W, 2)          
    
    # using torch grid sample to warp source image
    warped_image = f.grid_sample(source, deformation, mode=interp, padding_mode=padding, align_corners=align)
    warped_image = torch.clamp(warped_image, min=0, max=1) 
    
    return warped_image

