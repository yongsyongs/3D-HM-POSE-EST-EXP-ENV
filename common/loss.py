import torch
import numpy as np

# input: (..., dim)
def mpjpe(pred, target):
    assert pred.shape == target.shape
    return torch.mean(torch.norm(pred - target, dim=-1))

# input: (..., dim)
def w_mpjpe(pred, target, w):
    assert pred.shape == target.shape
    assert w.shape[0] == pred.shape[0]
    return torch.mean(w * torch.norm(pred - target, dim=-1))

# input: (..., dim)
def p_mpjpe(pred, target):
    assert pred.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(pred, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = pred - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    pred_aligned = a*np.matmul(pred, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(pred_aligned - target, axis=-1))

# input: (N, T, J, dim)
def n_mpjpe(pred, target):
    assert pred.shape == target.shape
    
    norm_pred = torch.mean(torch.sum(pred**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target*pred, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_pred
    return mpjpe(scale * pred, target)


# input: (..., T = time_axis, ..., dim)
def mpjve(pred, target, time_axis=0):
    assert pred.shape == target.shape

    diff_func, mean_func, norm_func = torch_diff, wrap_torch_axis(torch.mean, ()), wrap_torch_axis(torch.norm, None) \
        if type(pred) is torch.Tensor else np.diff, np.mean, np.linalg.norm
    
    velocity_pred = diff_func(pred, axis=time_axis)
    velocity_target = diff_func(target, axis=time_axis)
    
    return mean_func(norm_func(velocity_pred - velocity_target, axis=-1))

# input (..., T = time_axis, ..., dim)
def mpjae(pred, target, time_axis=0):
    assert pred.shape == target.shape
    
    diff_func, mean_func, norm_func = torch_diff, wrap_torch_axis(torch.mean, ()), wrap_torch_axis(torch.norm, None) \
        if type(pred) is torch.Tensor else np.diff, np.mean, np.linalg.norm


    acceleration_pred = diff_func(diff_func(pred, axis=time_axis), axis=time_axis)
    acceleration_target = diff_func(diff_func(target, axis=time_axis), axis=time_axis)

    return mean_func(norm_func(acceleration_pred - acceleration_target, axis=-1))

def wrap_torch_axis(func, axis_default):
    def f(*args, **kwargs, axis=axis_default):
        return func(*args, **kwargs, dim=axis)
    return f
    
def torch_diff(x, axis=0):
    input_dim = len(x.shape)

    end_slice = [None] * input_dim
    end_slice[axis] = slice(1, None, None)
    start_slice = [None] * input_dim
    start_slice[axis] = slice(None, -1, None)
    
    return x[end_slice] - x[start_slice]