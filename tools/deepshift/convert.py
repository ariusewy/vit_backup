from sys import modules
import torch
import torch.nn as nn
import numpy as np
import math
import copy

import deepshift.modules
import deepshift.modules_q
import deepshift.utils as utils
import deepshift.ste as ste
from quantize import quantize
from vit_pytorch_loc.vit_pytorch import Transformer, ViT,Attention,FeedForward

def update_B(model,updata_way,weight_tensor):
    for _,module in model._modules.items():
        if len(list(module.children())) > 0:
                update_B(module,'average',weight_tensor)
        if type(module)==deepshift.modules.LinearShift:
            head = int(module.oriweight.shape[0]/64)
            Origin = torch.chunk(module.oriweight,head,0)
            weight_ps = ste.unsym_grad_mul(2**module.shift.data, module.sign.data)
            Ces = torch.chunk(weight_ps,head,0)
            Bes=[]
            for i in range(head):
                Be,_ = torch.lstsq(Origin[i].t(),Ces[i].t())
                Bes.append((Be[:Ces[i].shape[0]]).t())

            Be_sum= torch.zeros([64,64],dtype=torch.float32,device=torch.device(Bes[1].device))
            if updata_way == 'average':
                for tensor in Bes:
                    Be_sum = Be_sum + tensor                    
                Be_sum_Q = Be_sum/head
                module.Be.data=quantize(Be_sum_Q, flatten_dims=(-1,), num_bits=8, dequantize=True)
            elif updata_way == 'head_score':
                factor = int(len(Bes)/len(weight_tensor))
                weight_tensor_new = weight_tensor.expand(factor,len(weight_tensor)).reshape(len(Bes),-1)/factor
                weight_tensor_new = torch.squeeze(weight_tensor_new)
                for  i in range(len(Bes)):
                    Be_sum = Be_sum + Bes[i]*weight_tensor_new[i]
                module.Be.data=quantize(Be_sum, flatten_dims=(-1,), num_bits=8, dequantize=True)

def update_Transformer(model,weight_tensor,updata_way):
    i=0
    for _,block in model._modules.items():
        atten = block._modules['0']._modules['fn']._modules['fn']
        fn = block._modules['1']._modules['fn']._modules['fn']
        if type(atten)==Attention:
            update_B(atten,updata_way,weight_tensor[i])
            i=i+1
        if type(fn)==FeedForward:
            update_B(fn,'average',weight_tensor[i-1])
        

            

def update_weight_tensor_old(model,weight_tensor,updata_way):
    #atten_layer=0
    for _,module in model._modules.items():
        if type(module)==Transformer:
            update_Transformer(module._modules['layers'],weight_tensor,updata_way)
        elif type(module)!=Transformer and len(list(module.children())) > 0:
            update_weight_tensor_old(module,weight_tensor,updata_way)

def update_weight_tensor(model,updata_way):
    for _,module in model._modules.items():
        if type(module)==Attention:
            weight_tensor = module.head_score
            update_B(module,updata_way,weight_tensor)
        if type(module)==deepshift.modules.LinearShift:
            if module.shift.shape[0]< 64:
                module.Be.data= module.Be.data
            else:
                head = int(module.oriweight.shape[0]/64)
                Origin = torch.chunk(module.oriweight,head,0)
                weight_ps = ste.unsym_grad_mul(2**module.shift.data, module.sign.data)
                Ces = torch.chunk(weight_ps,head,0)
                Bes=[]
                for i in range(head):
                    Be,_ = torch.lstsq(Origin[i].t(),Ces[i].t())
                    Bes.append((Be[:Ces[i].shape[0]]).t())

                Be_sum= torch.zeros([64,64],dtype=torch.float32,device=torch.device(Bes[1].device))
                for tensor in Bes:
                    Be_sum = Be_sum + tensor                    
                Be_sum_Q = Be_sum/head
                module.Be.data=quantize(Be_sum_Q, flatten_dims=(-1,), num_bits=8, dequantize=True)
        elif type(module)!=Attention and len(list(module.children())) > 0:
            update_weight_tensor(module,updata_way) 
       
    
def update_convert(model,weight_tensor,updata_way):
    i=0
    for param in model.parameters():
        if param.shape == torch.Size([64, 64]):
            print(param,weight_tensor,updata_way)


def convert_to_shift(model, shift_depth, shift_type, convert_all_linear=True, convert_weights=False, Use_B=False,freeze_sign = False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
    conversion_count = 0
    for name, module in reversed(model._modules.items()):# reverse the item
        if len(list(module.children())) > 0:
            model._modules[name], num_converted = convert_to_shift(model=module, shift_depth=shift_depth-conversion_count, shift_type=shift_type, convert_all_linear=convert_all_linear, convert_weights=convert_weights, freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                                   weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)
            conversion_count += num_converted
        if type(module) == nn.Linear and (convert_all_linear == True or conversion_count < shift_depth):
            linear = module
            if shift_type == 'Q':
                shift_linear = deepshift.modules_q.LinearShiftQ(module.in_features, module.out_features, module.bias is not None, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                                weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits) 
                shift_linear.weight = linear.weight
                if linear.bias is not None:
                    shift_linear.bias.data = utils.round_to_fixed(linear.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)

                if use_cuda==True and use_kernel == True:
                    shift_linear.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(linear.weight))
            elif shift_type == 'PS':
                shift_linear = deepshift.modules.LinearShift(module.weight,module.in_features, module.out_features, module.bias is not None, Use_B=Use_B,freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                             weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)

                if convert_weights == True:#加载原始权重并规则化，用SE之后的规则化
                    shift_linear.shift.data, shift_linear.sign.data = utils.get_shift_and_sign(linear.weight)
                    shift_linear.bias = linear.bias
                    if Use_B:
                        if linear.weight.shape[0]< 128:
                            shift_linear.Be.data= torch.eye(64)
                        else:
                            Origin = torch.chunk(linear.weight,int(linear.weight.shape[0]/64),0)
                            Ces = torch.chunk(torch.mul(2**shift_linear.shift.data,shift_linear.sign.data),int(shift_linear.shift.data.shape[0]/64),0)
                            Bes=[]
                            for i in range(int(linear.weight.shape[0]/64)):
                                BeT,_ = torch.lstsq(Origin[i].t(),Ces[i].t())
                                Bes.append((BeT[:Ces[i].shape[0]]).t())
                            Be_sum= torch.zeros([64,64],dtype=torch.float32,device=torch.device(Bes[1].device))
                            for tensor in Bes:
                                Be_sum = Be_sum + tensor                    
                            Be_sum_Q = Be_sum/int(linear.weight.shape[0]/64)                        
                            shift_linear.Be.data=quantize(Be_sum_Q, flatten_dims=(-1,), num_bits=8, dequantize=True)
                    if use_cuda==True and use_kernel == True:
                        shift_linear.conc_weight = utils.compress_bits(shift_linear.shift.data, shift_linear.sign.data)
            else:
                raise ValueError('Unsupported shift_type argument: ', shift_type)

            model._modules[name] = shift_linear
            if convert_all_linear == False:
                conversion_count += 1

        if type(module) == nn.Conv2d and conversion_count < shift_depth:
            conv2d = module

            if shift_type == 'Q':
                shift_conv2d = deepshift.modules_q.Conv2dShiftQ(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                module.padding, module.dilation, module.groups,
                                                module.bias is not None, module.padding_mode, 
                                                use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits) 
                shift_conv2d.weight = conv2d.weight
                if conv2d.bias is not None:
                    shift_conv2d.bias.data = utils.round_to_fixed(conv2d.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)

                if use_cuda==True and use_kernel == True:
                    shift_conv2d.conc_weight = utils.compress_bits(*utils.get_shift_and_sign(conv2d.weight))

            elif shift_type == 'PS':
                shift_conv2d = deepshift.modules.Conv2dShift(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                                module.padding, module.dilation, module.groups,
                                                module.bias is not None, module.padding_mode,
                                                freeze_sign=freeze_sign, use_kernel=use_kernel, use_cuda=use_cuda, rounding=rounding, 
                                                weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)

                if convert_weights == True:
                    shift_conv2d.shift.data, shift_conv2d.sign.data = utils.get_shift_and_sign(conv2d.weight)
                    shift_conv2d.bias = conv2d.bias

                if use_cuda==True and use_kernel == True:
                    shift_conv2d.conc_weight = utils.compress_bits(shift_conv2d.shift.data, shift_conv2d.sign.data)
               
            model._modules[name] = shift_conv2d
            conversion_count += 1

    return model, conversion_count

def round_shift_weights(model, clone=False, weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
    if(clone):
        model = copy.deepcopy(model)

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = round_shift_weights(model=module, weight_bits=weight_bits, act_integer_bits=act_integer_bits, act_fraction_bits=act_fraction_bits)

        if type(module) == deepshift.modules.LinearShift or type(module) == deepshift.modules.Conv2dShift:
            module.shift.data = module.shift.round()
            module.sign.data = module.sign.round().sign()

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)
        elif type(module) == deepshift.modules_q.LinearShiftQ or type(module) == deepshift.modules_q.Conv2dShiftQ:
            module.weight.data = utils.clampabs(module.weight.data, 2**module.shift_range[0], 2**module.shift_range[1]) 
            module.weight.data = utils.round_power_of_2(module.weight)

            if (module.bias is not None):
                module.bias.data = utils.round_to_fixed(module.bias, integer_bits=act_integer_bits, fraction_bits=act_fraction_bits)

    return model

def count_layer_type(model, layer_type):
    count = 0
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # recurse
            count += count_layer_type(model=module, layer_type=layer_type)
        if type(module) == layer_type:
            count += 1

    return count    