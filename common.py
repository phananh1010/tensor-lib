import torch as tc

def img2patches(x, kernel):
    #spatially divide x into patches of equal size based on kernel
    xb, xc, xh, xw     = x.shape
    kernel_h, kernel_w = kernel
    step_h, step_w     = kernel
    tmp                = x.unfold(2, kernel_h, step_h).unfold(3, kernel_w, step_w).permute(2, 3, 0, 1, 4, 5)
    windows            = tmp.reshape(-1, xc, kernel_h, kernel_w)
    return windows

def patches2img(windows, xshape):
    #TODO: convert patches back to large images
    wb, wc, wh, ww     = windows.shape
    xb, xc, xh, xw     = xshape
    tmp = patches.reshape(xh//ph, xw//pw, 1, pc, ph, pw).permute(2, 3, 0, 4, 1, 5)
    x1  = tmp.reshape(xb, xc, xh, xw)
    return x1