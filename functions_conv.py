import torch
from dezero.core import Function, as_variable
from dezero.utils import pair, get_conv_outsize
import torch.nn.functional as F_torch
from dezero import Function, Variable


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---------------------------------------------------------------------
# im2col / col2im
# ---------------------------------------------------------------------
def im2col_array(img, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    img_padded = torch.nn.functional.pad(img, (PW, PW, PH, PH))
    col = torch.zeros((N, C, KH, KW, OH, OW), device=img.device)

    for y in range(KH):
        y_max = y + SH * OH
        for x in range(KW):
            x_max = x + SW * OW
            col[:, :, y, x, :, :] = img_padded[:, :, y:y_max:SH, x:x_max:SW]

    if to_matrix:
        col = col.permute(0, 4, 5, 1, 2, 3).reshape(N * OH * OW, -1)
    return col


def col2im_array(col, img_shape, kernel_size, stride, pad, to_matrix=True):
    N, C, H, W = img_shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)

    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    if to_matrix:
        col = col.reshape(N, OH, OW, C, KH, KW).permute(0, 3, 4, 5, 1, 2)

    img = torch.zeros((N, C, H + 2*PH + SH - 1, W + 2*PW + SW - 1), device=col.device)
    for y in range(KH):
        y_max = y + SH * OH
        for x in range(KW):
            x_max = x + SW * OW
            img[:, :, y:y_max:SH, x:x_max:SW] += col[:, :, y, x, :, :]

    return img[:, :, PH:H+PH, PW:W+PW]

# ---------------------------------------------------------------------
# Simple conv2d / pooling
# ---------------------------------------------------------------------
def conv2d_simple(x, W, b=None, stride=1, pad=0):
    x, W = as_variable(x), as_variable(W)
    N, C, H, W_ = x.shape
    OC, _, KH, KW = W.shape
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W_, KW, SW, PW)

    col = im2col_array(x.data, (KH, KW), stride, pad, to_matrix=True)
    W_col = W.data.view(OC, -1).t()
    out = col @ W_col
    if b is not None:
        out += b.data
    out = out.view(N, OH, OW, OC).permute(0, 3, 1, 2)
    return as_variable(out)


def pooling_simple(x, kernel_size, stride=1, pad=0):
    x = as_variable(x)
    N, C, H, W = x.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    col = im2col_array(x.data, (KH, KW), stride, pad, to_matrix=False)
    col = col.view(N, C, KH * KW, OH, OW)
    out, _ = col.max(dim=2)
    return as_variable(out)


# ---------------------------------------------------------------------
# im2col / col2im as Function
# ---------------------------------------------------------------------
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix=True):
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        return im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)

    def backward(self, gy):
        return col2im_array(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)


def im2col(x, kernel_size, stride=1, pad=0, to_matrix=True):
    return Im2col(kernel_size, stride, pad, to_matrix)(x)


class Col2im(Function):
    def __init__(self, input_shape, kernel_size, stride, pad, to_matrix=True):
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        return col2im_array(x, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

    def backward(self, gy):
        return im2col(gy, self.kernel_size, self.stride, self.pad, self.to_matrix)


def col2im(x, input_shape, kernel_size, stride=1, pad=0, to_matrix=True):
    return Col2im(input_shape, kernel_size, stride, pad, to_matrix)(x)


# dezero/functions_conv.py 내부에 넣기 (또는 임시로 별도 정의해서 테스트 가능)



class Conv2dSimple(Function):
    def __init__(self, stride=1, pad=0):
        self.stride = stride
        self.pad = pad

    def forward(self, x, W):
        self.x = x  # Save for backward
        self.W = W
        self.x_shape = x.shape
        self.W_shape = W.shape

        N, C, H, W_ = x.shape
        OC, _, KH, KW = W.shape

        # im2col: unfold
        self.col = F_torch.unfold(
            x, kernel_size=(KH, KW), stride=self.stride, padding=self.pad
        )  # (N, C*KH*KW, OH*OW)
        self.col_W = W.reshape(OC, -1)  # (OC, C*KH*KW)

        out = torch.matmul(self.col_W, self.col)  # (N, OC, OH*OW)

        OH = (H + 2 * self.pad - KH) // self.stride + 1
        OW = (W_ + 2 * self.pad - KW) // self.stride + 1

        out = out.view(N, OC, OH, OW)  # (N, OC, OH, OW)
        return out

    def backward(self, gy):
        # DeZero Variable → torch.Tensor
        gy = gy.data
        x, W = self.x, self.W
        N, C, H, W_ = self.x_shape
        OC, _, KH, KW = self.W_shape

        OH = (H + 2 * self.pad - KH) // self.stride + 1
        OW = (W_ + 2 * self.pad - KW) // self.stride + 1

        gy = gy.contiguous().view(N, OC, OH * OW)

        # ∂L/∂W
        gW = torch.matmul(gy, self.col.transpose(1, 2))  # (N, OC, C*KH*KW)
        gW = gW.sum(dim=0).view(self.W_shape)  # (OC, C, KH, KW)

        # ∂L/∂x
        col_W_T = self.col_W.t()  # (C*KH*KW, OC)
        gcol = torch.matmul(col_W_T, gy)  # (N, C*KH*KW, OH*OW)

        gx = F_torch.fold(
            gcol,
            output_size=(H, W_),
            kernel_size=(KH, KW),
            stride=self.stride,
            padding=self.pad,
        )  # (N, C, H, W)

        return Variable(gx), Variable(gW)

    
    

   

def conv2d_simple(x, W, stride=1, pad=0):
    return Conv2dSimple(stride, pad)(x, W)


