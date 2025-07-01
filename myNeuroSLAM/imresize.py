import numpy as np


def deriveSizeFromScale(img_shape, scale):
    return np.ceil(np.array(img_shape) * np.array(scale)).astype(int)


def deriveScaleFromSize(img_shape_in, img_shape_out):
    return np.array(img_shape_out) / np.array(img_shape_in)


def triangle(x):
    x = np.clip(x, -1, 1).astype(np.float64)
    return (1 - np.abs(x)) * ((np.abs(x) <= 1).astype(np.float64))


def cubic(x):
    x = np.abs(x).astype(np.float64)
    p = (1.5 * x**3 - 2.5 * x**2 + 1) * (x <= 1)
    q = (-0.5 * x**3 + 2.5 * x**2 - 4 * x + 2) * ((1 < x) & (x <= 2))
    return p + q


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2).astype(int)
    P = int(np.ceil(kernel_width)) + 2
    indices = left[:, None] + np.arange(P) - 1
    weights = h(u[:, None] - indices - 1)
    weights /= np.sum(weights, axis=1, keepdims=True)
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    outimg = 0
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, output_shape=None, scalar_scale=None, method='bicubic', mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        raise ValueError('unidentified kernel method supplied')
    
    kernel_width = 4.0
    # Fill scale and output_size
    if scalar_scale is not None and output_shape is not None:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    if scalar_scale is not None:
        scalar_scale = np.float(scalar_scale)
        scale = [scalar_scale, scalar_scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif output_shape is not None:
        scale = deriveScaleFromSize(I.shape, output_shape)
        output_size = list(output_shape)
    else:
        raise ValueError('either scalar_scale OR output_shape should be defined')
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return np.float32(B)


def convertDouble2Byte(I):
    B = np.clip(I, 0.0, 1.0)
    B = 255 * B
    return np.around(B).astype(np.uint8)
