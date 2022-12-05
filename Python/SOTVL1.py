import torch, math
import torch_dct_bak as dct 
import timeit


def central_finite_diff(image):
    """ image size (N, 1, H, W) """
    h, w = image.shape[-2:]
    
    C1 = torch.roll(image, -1, -1)
    C1[...,w-1] = C1[...,w-2]
    
    C2 = torch.roll(image,  1, -1)
    C2[...,0] = C2[...,1]
    
    C3 = torch.roll(image, -1, -2)
    C3[...,h-1,:] = C3[...,h-2,:]
    
    C4 = torch.roll(image,  1, -2)
    C4[...,0,:] = C4[...,1,:]
    
    image_x = (C1 - C2)/2
    image_y = (C3 - C4)/2
    
    return image_x, image_y


def Fx(u):
    """ u size (N, H, W) forward difference x"""
    n, h, w = u.shape
    Fxu = torch.roll(u, -1, -1) - u 
    Fxu[...,w-1] = torch.zeros(n, h)
    
    return Fxu
    

def Fy(u):
    """ u size (N, H, W) forward difference y"""
    n, h, w = u.shape
    Fyu = torch.roll(u, -1, -2) - u 
    Fyu[...,h-1,:] = torch.zeros(n, w)
    
    return Fyu


def Bx(u):
    """ u size (N, H, W) backfward difference x"""
    w = u.shape[-1]
    Bxu = u - torch.roll(u, 1, -1) 
    Bxu[...,0] = u[...,0]
    Bxu[...,w-1] = -u[...,w-2]
    
    return Bxu

        
def By(u):
    """ u size (N, H, W) backfward difference y"""
    h = u.shape[-2]
    Byu = u - torch.roll(u, 1, -2) 
    Byu[...,0,:] = u[...,0,:]
    Byu[...,h-1,:] = -u[...,h-2,:]
    
    return Byu


def resolvent_operator(tmp1, tmp2, u0, v0, Ix, Iy, It, J11, J22, theta_2):
    
    rho_w = Ix * (tmp1 - u0) + Iy * (tmp2 - v0) + It
    zhat = theta_2 * rho_w / (J11 + J22 + 2.220446049250313e-16)
    tmp = zhat / torch.clamp(torch.abs(zhat), min=1)
    u = tmp1 - (Ix / theta_2) * tmp
    v = tmp2 - (Iy / theta_2) * tmp
   
    return u, v


def direct_dct_solver(u, d, w1, w2, w3, b1, b2, b3, G, theta_w, theta_tv, solver='fft'):
    
    div_w_b = Bx(Fx(w1-b1)) + By(Fy(w2-b2)) + 2*By(Bx(w3-b3))
    g = theta_w*(u+d) + theta_tv*div_w_b
    
    if solver == 'dct':
        W = dct.idct_2d(dct.dct_2d(g) / (theta_w + theta_tv*G**2))
    else:
        W = torch.irfft(torch.div(torch.rfft(g, 2, onesided=False), theta_w + theta_tv*G**2), 2, onesided=False)
    
    return W


def vector_soft_threshold(Wx, Wy, b11, b12, b13, b21, b22, b23, lmbda, theta_tv):
       
    c11 = Bx(Fx(Wx)) + b11
    c12 = By(Fy(Wx)) + b12
    c13 = Fy(Fx(Wx)) + b13
    c21 = Bx(Fx(Wy)) + b21
    c22 = By(Fy(Wy)) + b22
    c23 = Fy(Fx(Wy)) + b23
      
    abs_c = torch.sqrt(c11**2 + c12**2 + 2*c13**2 + c21**2 + c22**2 + 2*c23**2 + 2.220446049250313e-16)
    
    w11 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c11 / abs_c
    w12 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c12 / abs_c
    w13 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c13 / abs_c
    w21 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c21 / abs_c
    w22 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c22 / abs_c
    w23 = torch.clamp(abs_c - lmbda/theta_tv, min=0) * c23 / abs_c
    
    return w11, w12, w13, w21, w22, w23, c11, c12, c13, c21, c22, c23


def sotvl1_admm_optimizer(im1, im2, disp, lmbda=0.1, theta_w=.1, theta_tv=1, 
                          tol=1e-3, maxIter=500, acc=True, alpha=1.8, 
                          solver='fft', device='cpu'):
    
    im1, im2 = im1.squeeze(1), im2.squeeze(1) 
    Ix, Iy = central_finite_diff(im1)
    It = im1 - im2
    J11 = Ix*Ix
    J22 = Iy*Iy
        
    N, H, W = im1.shape
    u0, v0 = disp[:,1], disp[:,0]
    b11 = torch.zeros(N, H, W, device=torch.device(device))
    b12 = torch.zeros(N, H, W, device=torch.device(device))
    b13 = torch.zeros(N, H, W, device=torch.device(device))
    b21 = torch.zeros(N, H, W, device=torch.device(device))
    b22 = torch.zeros(N, H, W, device=torch.device(device))
    b23 = torch.zeros(N, H, W, device=torch.device(device))
    w11 = torch.zeros(N, H, W, device=torch.device(device))
    w12 = torch.zeros(N, H, W, device=torch.device(device))
    w13 = torch.zeros(N, H, W, device=torch.device(device))
    w21 = torch.zeros(N, H, W, device=torch.device(device))
    w22 = torch.zeros(N, H, W, device=torch.device(device))
    w23 = torch.zeros(N, H, W, device=torch.device(device))
    d1  = torch.zeros(N, H, W, device=torch.device(device))
    d2  = torch.zeros(N, H, W, device=torch.device(device))
    Wx  = torch.zeros(N, H, W, device=torch.device(device))
    Wy  = torch.zeros(N, H, W, device=torch.device(device))
    u   = torch.zeros(N, H, W, device=torch.device(device))
    v   = torch.zeros(N, H, W, device=torch.device(device))
    
    X, Y = torch.meshgrid(torch.linspace(0, H-1, H, device=torch.device(device)), 
                          torch.linspace(0, W-1, W, device=torch.device(device)))
    
    if solver == 'dct': 
        G = -2 * (torch.cos(math.pi * X/H) + torch.cos(math.pi * Y/W) - 2)
        G = G.unsqueeze(0).repeat(N, 1, 1)
    else: 
        G = -2 * (torch.cos(2 * math.pi * X/H) + torch.cos(2 * math.pi * Y/W) - 2)
        G = G.unsqueeze(0).unsqueeze(-1).repeat(N, 1, 1, 2)
      
        
    start = timeit.default_timer()        
    for i in range(maxIter):
                
        #######################################################################
        temp_u = u
        temp_v = v     
        
        #######################################################################
        u, v = resolvent_operator(Wx-d1, Wy-d2, u0, v0, Ix, Iy, It, J11, J22, theta_w)
        
        #######################################################################
        if acc:
            u = alpha * u + (1-alpha) * Wx
            v = alpha * v + (1-alpha) * Wy
        
        #######################################################################
        Wx = direct_dct_solver(u, d1, w11, w12, w13, b11, b12, b13, G, theta_w, theta_tv, solver)
        Wy = direct_dct_solver(v, d2, w21, w22, w23, b21, b22, b23, G, theta_w, theta_tv, solver)
 
        #######################################################################
        w11, w12, w13, w21, w22, w23, c11, c12, c13, c21, c22, c23 = vector_soft_threshold(Wx, Wy, b11, b12, b13, b21, b22, b23, lmbda, theta_tv)
           
        #######################################################################
        b11, b12, b13, b21, b22, b23, d1, d2 = c11-w11, c12-w12, c13-w13, c21-w21, c22-w22, c23-w23, d1+u-Wx, d2+v-Wy
           
        #######################################################################
        stop1 = torch.sum(torch.abs(u - temp_u)) / torch.sum(torch.abs(temp_u) + 2.220446049250313e-16)
        stop2 = torch.sum(torch.abs(v - temp_v)) / torch.sum(torch.abs(temp_v) + 2.220446049250313e-16)       
        
        end = timeit.default_timer()
        if stop1 < tol and stop2 < tol and i > 2:
            # print('   iterate {} times costs {:.3f}s, stop due to converge to tolerance {}\n'.format(i, end-start, tol));
            break # set tol to break iterion       
            
    return torch.stack((v, u), 1)