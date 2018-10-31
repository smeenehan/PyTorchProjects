import copy
import numpy as np
import torch
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import grad

def get_eb_toggle():
    """Return a fucntion which toggles the PyTorch nn backend between using the 
    standard implementations and the Excitation Backprop enabled versions defined 
    in this file. Note that this is accomplished by replacing the definitions in
    torch.nn.functional. This is sufficient to override higher-level modules (e.g.,
    nn.Conv2D), but does not affect the lower level CUDA wrappers (e.g.,
    torch.conv2d)"""
    orig_torch_fns = {}
    orig_torch_fns['linear'] = copy.deepcopy(torch.nn.functional.linear)
    orig_torch_fns['conv2d'] = copy.deepcopy(torch.nn.functional.conv2d)
    orig_torch_fns['avg_pool2d'] = copy.deepcopy(torch.nn.functional.avg_pool2d)
    eb_on = False

    def toggle():
        nonlocal eb_on
        if eb_on is False:
            eb_on = True
            torch.nn.functional.linear = EBLinear.apply
            torch.nn.functional.conv2d = EBConv2d.apply
            torch.nn.functional.avg_pool2d = EBAvgPool2d.apply
        else:
            eb_on = False
            torch.nn.functional.linear = orig_torch_fns['linear']
            torch.nn.functional.conv2d = orig_torch_fns['conv2d']
            torch.nn.functional.avg_pool2d = orig_torch_fns['avg_pool2d']

    return toggle

def excitation_backprop(model, input, output_prob, target_layer=0, contrastive=False):
    """Perform excitation backprop on a set of input images, according to a specified
    probability distribution.

    Parameters
    ----------
    model : Module
        Image classifier model.
    input : Tensor
        Input tensor, in NCHW format.
    output_prob : Tensor
        Desired output probability distribution. Must be equal in size to the output
        of model, and be a valid probability distribution (positive, sums to 1).
    target_layer : int, optional
        Layer at which we wish to compute the excitation response. Note that this
        is an index in the flattened list of Modules in model, so care must be
        taken when specifying this for branching models like ResNet or Inception.
    contrastive : bool, default
        Whether or not to use contrastive excitation backprop (i.e., desired 
        probability minus the dual).
    """
    input.requires_grad = True
    model.eval()

    all_modules = recursive_module_list(model)
    top_layer = all_modules[-1]
    target_layer = all_modules[target_layer]

    # set a forward hook to track the output tensor of the top layer
    top_output, target, contrast_output = None, None, None
    def top_hook(m, i, o): nonlocal top_output; top_output = o
    def target_hook(m, i, o): nonlocal target; target = i[0]
    def contrast_hook(m, i, o): nonlocal contrast_output; contrast_output = i[0]

    hook_1 = top_layer.register_forward_hook(top_hook)
    hook_2 = target_layer.register_forward_hook(target_hook)
    hook_3 = top_layer.register_forward_hook(contrast_hook)

    _ = model(input)
    hook_1.remove(); hook_2.remove(); hook_3.remove()

    if not contrastive:
        return torch.autograd.grad(top_output, target, grad_outputs=output_prob)[0]

    pos_evidence = torch.autograd.grad(top_output, contrast_output, 
                                       grad_outputs=output_prob.clone(),
                                       retain_graph=True)[0]
    top_layer.weight.data *= -1.0
    neg_evidence = torch.autograd.grad(top_output, contrast_output, 
                                       grad_outputs=output_prob.clone(),
                                       retain_graph=True)[0]
    top_layer.weight.data *= -1.0
    contrastive_signal = pos_evidence-neg_evidence
    return torch.autograd.grad(contrast_output, target, grad_outputs=contrastive_signal)[0]


def recursive_module_list(parent, module_list=None):
    """Recusively generate a flattened list of all modules within a PyTorch 
    Module."""
    module_list = [] if module_list is None else module_list
    children = list(parent.children())
    if len(children) > 0:
        for idx, child in enumerate(children):
            module_list += recursive_module_list(child, module_list=[])
    else:
        module_list.append(parent)
    return module_list

class EBLinear(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None

        # zero out "inhibitory neurons" (negative weights), ensure inputs are
        # zero-referred, and normalize inflowing gradient so that grad_input becomes
        # a probability distribution
        weight_plus = weight.clamp(min=0)
        input_min = input.data.min()
        if input_min < 0:
            input.data -= input_min
        norm_grad = grad_output/(input.matmul(weight_plus.t()) + 1e-10)

        if ctx.needs_input_grad[0]:
            grad_input = norm_grad.matmul(weight_plus)
            grad_input *= input

        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class EBConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        return torch.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        grad_input = grad_weight = grad_bias = None

        # zero out "inhibitory neurons" (negative weights), ensure inputs are
        # zero-referred, and normalize inflowing gradient so that grad_input becomes
        # a probability distribution
        weight_plus = weight.clamp(min=0)
        input_min = input.data.min()
        if input_min < 0:
            input.data -= input_min
        norm_factor = torch.conv2d(input, weight_plus, None, stride, padding, 
                                   dilation,groups)
        norm_grad_output = grad_output/(norm_factor + 1e-10)

        if ctx.needs_input_grad[0]:
            grad_input = grad.conv2d_input(
                input.size(), weight_plus, norm_grad_output, stride=stride, 
                padding=padding, dilation=dilation, groups=groups)
            grad_input *= input

        if ctx.needs_input_grad[1]:
            grad_weight = grad.conv2d_weight(
                input, weight.size(), grad_output, stride=stride, 
                padding=padding, dilation=dilation, groups=groups)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = torch.sum(grad_output, (0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None

class EBAvgPool2d(Function):
    """Note that our implentation is phrased as a 2D convolution. The advantage is
    that we can keep the re-implementation in Python while still leveraging 
    the existing efficient PyTorch implementations in CUDA. In particular, the 
    custom backward pass is simplified greatly by being able to use 
    torch.nn.grad.conv2d_input.
    """
    @staticmethod
    def forward(ctx, input, kernel_size, stride=None, padding=0, ceil_mode=False,
                count_include_pad=True):
        ctx.stride = _pair(stride if stride is not None else kernel_size)
        ctx.padding = padding
        _, C, _, _ = input.size()
        new_kernel_size = _pair(kernel_size)
        N_avg = new_kernel_size[0]*new_kernel_size[1]
        weight = torch.zeros((C, C, new_kernel_size[0], new_kernel_size[1]))
        for idx in range(C):
            weight[idx, idx, :] = 1/N_avg
        ctx.weight = weight
        output = torch.conv2d(input, weight, None, ctx.stride, ctx.padding, 1, 1)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_variables
        weight, stride, padding = ctx.weight, ctx.stride, ctx.padding
        grad_input = None

        input_min = input.data.min()
        if input_min < 0:
            input.data -= input_min
        norm_grad_output = grad_output/(output + 1e-10)
        norm_grad_output *= (output>0).float()

        if ctx.needs_input_grad[0]:
            grad_input = grad.conv2d_input(
                input.size(), weight, norm_grad_output, stride=stride, 
                padding=padding, dilation=1, groups=1)
            grad_input *= input
        return grad_input, None, None, None, None, None