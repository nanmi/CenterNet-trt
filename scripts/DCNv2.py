import torch
import torch.nn as nn
import math
from torchvision.ops.deform_conv import deform_conv2d
from torch.nn.modules.utils import _pair
from torch.autograd import Function

# # import the DCN plugin
# from torch.utils.cpp_extension import load

# import os
# abs_path = os.path.dirname(os.path.realpath(__file__))
# sources=['/src/cuda/dcn_v2_cuda.cu', '/src/cuda/dcn_v2_im2col_cuda.cu', '/src/cpu/dcn_v2_cpu.cpp', '/src/cpu/dcn_v2_im2col_cpu.cpp']
# sources = [abs_path+file for file in sources]
# dcn_plugin = load(name='dcn_plugin', sources=sources)


# define the new-style autograd function
class DCNv2Function(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias=None, stride=1, padding=1, dilation=1, deformable_groups=1):
        # output = dcn_plugin.dcn_v2_cuda_forward(input, weight, bias, offset, mask, 3, 3, stride, stride, padding, padding, dilation, dilation, deformable_groups)
        output = deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask)
        ctx.save_for_backward(input, offset, weight, mask, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.deformable_groups = deformable_groups
        return output

    # @staticmethod
    # def backward(ctx, grad_output):
    #     input, offset, weight, bias = ctx.saved_tensors
    #     grad_input, grad_offset, grad_weight, grad_bias = dcn_plugin.dcn_v2_backward(grad_output, input, weight, offset, bias, ctx.intermediate, ctx.stride, ctx.padding, ctx.dilation, ctx.deformable_groups)
    #     return grad_input, grad_offset, grad_weight, grad_bias, None, None, None, None


class DCNv2(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        func = DCNv2Function.apply
        return func(input, offset, mask, self.weight, self.bias)


class DCN(DCNv2):

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride, padding,
                 dilation=1, deformable_groups=1):
        super(DCN, self).__init__(in_channels, out_channels,
                                  kernel_size, stride, padding, dilation, deformable_groups)

        self.conv_offset_mask = nn.Conv2d(self.in_channels,
                                          self.deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
                                          kernel_size=self.kernel_size,
                                          stride=(self.stride, self.stride),
                                          padding=(self.padding, self.padding),
                                          bias=True)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        # 1. method 1
        # o1, o2, mask = torch.chunk(out, 3, dim=1)
        # 2. method 2
        [o1, o2, mask] = torch.split(out, int(out.shape[1]/3), dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # func = DCNv2Function(self.stride, self.padding, dilation=self.dilation, deformable_groups=self.deformable_groups)
        func = DCNv2Function.apply
        return func(input, offset, mask, self.weight, self.bias)


from torch.onnx.symbolic_helper import parse_args
def dcn_v2_symbolic(g, input, offset, weight, bias=None, stride=1, padding=1, dilation=1, mask=None):
    # map the PyTorch Function to the ONNX operator
    return g.op('torchvision::deform_conv2d', input, offset, weight, bias, stride_i=stride, padding_i=padding, dilation_i=dilation, mask=mask)

# Register the 'deform_conv2d' operator
opset_version = 13  # Replace with the desired opset version
torch.onnx.register_custom_op_symbolic('torchvision::deform_conv2d', dcn_v2_symbolic, opset_version)


if __name__ == "__main__":
    input = torch.zeros([1, 64, 512, 512]).cuda()
    model = DCN(64, 64, kernel_size=(3,3), stride=1, padding=1, dilation=1, deformable_groups=1)
    torch.onnx.export(model, input, "dcn.onnx", operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH, opset_version=13)

