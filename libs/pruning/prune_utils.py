import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

from libs.pruning.gates import virtual_gate

from libs.models.AttentionUnet3D import init_weights

def freeze_weights(model):
    """
        Pruning을 위한 gates들을 제외하고 나머지 parameter들을 고정
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        elif isinstance(m, nn.Conv3d):
            m.weight.requires_grad = False
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.requires_grad = False
            m.bias.requires_grad = False
        elif isinstance(m, nn.Conv2d):
            m.weight.requires_grad = False


def count_structure_groupid(model):
    """ 
        모델의 structure(layer별 filter 수)와 group 정보(어떤 layer들이 skip connection에 의해 하나의 group으로 묶이는지)를 확인하기 위한 함수 

        Returns:
            total_num_filters           : 모델안 전체 convolution filter의 수
            structure                   : layer별 filter 수
            group_ids                   : 모델 안 group들의 id list (하나의 group안 gate들은 동일한 id를 갖음)
            no_prune_idx                : 경량화를 진행하지 않는 layer index
    """
    structure = []
    group_ids = []
    no_prune_idx = []
    for m in model.modules():
        if isinstance(m, virtual_gate):
            if not m.prune:
                no_prune_idx.append(len(structure))
            structure.append(m.width)
            group_ids.append(m.group_id)
    total_num_filters = sum(structure)
    return total_num_filters, structure, group_ids, no_prune_idx


def set_virtual_gate(model, vector):
    """ 
        Hyper network에서 gate들의 출력값을 받아 모델에 삽입된 virtual_gate들에게 전달
        모델안 virtual_gate의 gate_f class변수의 값을 gate들의 출력값으로 설정해주는 함수
        model forwarding시 Convolution layer에 의해 생성된 feature map은 virtual gate의 gate_f와 연산 후 다음 layer에 전달
        
        Args:
            model           : 
            vector          : gate들의 출력값 (soft or hard)
    """
    i = 0
    for m in model.modules():
        if isinstance(m, virtual_gate):
            m.set_structure_value(vector[i].unsqueeze(0))
            i += 1


def get_inchannel_idx(model_name, model):
    """
        Convolution layer의 입력 feature map이 어느 layer의 출력되었는지 matching 해주는 함수

        Returns:
            inchannel_idx           : 해당 index의 Convolution layer의 입력 feature map이 어느 layer index에서 출력되었는지 
    """

    module_name_list = []
    if isinstance(model, torch.nn.DataParallel):
        named_modules = list(model.module.named_modules())
    else:
        named_modules = list(model.named_modules())
    for n, m in named_modules:
        if isinstance(m, nn.Conv3d):
            module_name_list.append(n)
    
    inchannel_idx = [-1]    # 첫번째 inchannel size는 영상의 channel

    if model_name == 'AttentionUnet3D':
        
        encoder_output_idx = [1, 3, 5, 7]               # conv1, conv2, conv3, conv4
        mid_output_idx = [9, 10]                        # center, gating
        # decoder_output_idx = [45, 43, 41, 39]
        decoder_output_idx = [45, 43, 32, 21]           # up1, up2, up3, up4

        for i in range(1, len(module_name_list)):
            module_name = module_name_list[i].split('.')
            if 'attentionblock' in module_name[0]:
                gate_num = int(module_name[0][-1])
                
                if module_name[1] == 'combine_gates':
                    inchannel_idx.append([i-5, i-1])
                
                else:
                    if module_name[2] == 'theta' or module_name[2] == 'W':
                        inchannel_idx.append([encoder_output_idx[gate_num-1]])
                
                    elif module_name[2] == 'phi':
                        if gate_num == 4: 
                            inchannel_idx.append([mid_output_idx[1]])
                        else:
                            inchannel_idx.append([decoder_output_idx[gate_num]])
                    else:
                        inchannel_idx.append([i-1])
            elif 'up_concat' in module_name[0]:
                gate_num = int(module_name[0][-1])

                if module_name[2] == 'conv1':
                    if gate_num == 4:
                        inchannel_idx.append([i-1, mid_output_idx[0]])
                    elif gate_num == 1:
                        inchannel_idx.append([encoder_output_idx[0], decoder_output_idx[gate_num]])
                    else:
                        inchannel_idx.append([i-1, decoder_output_idx[gate_num]])
                else:
                    inchannel_idx.append([i-1])
            
            elif 'dsv' in module_name[0]:
                gate_num = int(module_name[0][-1])
                
                inchannel_idx.append([decoder_output_idx[gate_num-1]])
            
            elif module_name[0] == 'final':
                inchannel_idx.append([i-1, i-2, i-3, i-4])
            
            else:
                inchannel_idx.append([i-1])
        
    return inchannel_idx



def melt(model_name, model, hard_vector, new_model=None):
    """
        Pruning결과를 반영하여 실질적으로 모델 내 필터들을 제거하는 함수
    """

    new_modules = []
    outchannel_sizes = [int(v.sum().item()) for v in hard_vector]
    inchannel_idx = get_inchannel_idx(model_name=model_name, model=model)

    def _melt(modules, layer_id):
        keys = list(modules.keys())
        for i, k in enumerate(keys):
            if modules[k] == None:
                continue
            
            if len(modules[k]._modules) > 0:
                layer_id = _melt(modules[k]._modules, layer_id)
            
            if isinstance(modules[k], nn.Conv3d):
                if layer_id == 0:
                    cin = 1
                    in_mask = torch.ones(cin, dtype=torch.float)
                else:
                    cin = sum(outchannel_sizes[idx] for idx in inchannel_idx[layer_id])
                    in_mask = torch.cat([hard_vector[idx] for idx in inchannel_idx[layer_id]])

                cout = outchannel_sizes[layer_id]
                out_mask = hard_vector[layer_id]

                conv_replacer = nn.Conv3d(in_channels=cin, out_channels=cout, kernel_size=modules[k].kernel_size,
                                          stride=modules[k].stride, padding=modules[k].padding, groups=modules[k].groups,
                                          bias=False if getattr(modules[k], 'bias') is None else True).to(modules[k].weight.device)
                conv_replacer.weight.set_(modules[k].weight[out_mask==1][:, in_mask==1])
                
                if getattr(modules[k], 'bias') is not None:
                    conv_replacer.bias.set_(modules[k].bias[out_mask==1])
                
                if new_model is None:
                    modules[k] = conv_replacer
                else:
                    new_modules.append(conv_replacer)
                
                layer_id += 1

            elif isinstance(modules[k], nn.BatchNorm3d):
                bn_layer_id = layer_id - 1
                cout = outchannel_sizes[bn_layer_id]
                out_mask = hard_vector[bn_layer_id]

                bn_replacer = nn.BatchNorm3d(num_features=cout, eps=modules[k].eps, momentum=modules[k].momentum, affine=modules[k].affine,
                                             track_running_stats=modules[k].track_running_stats).to(modules[k].weight.device)
                bn_replacer.running_var.set_(modules[k].running_var[out_mask==1])
                bn_replacer.running_mean.set_(modules[k].running_mean[out_mask==1])
                bn_replacer.weight.set_(modules[k].weight[out_mask==1])
                bn_replacer.bias.set_(modules[k].bias[out_mask==1])

                if new_model is None:
                    modules[k] = bn_replacer
                else:
                    new_modules.append(bn_replacer)
            
            elif isinstance(modules[k], nn.InstanceNorm3d):
                in_layer_id = layer_id - 1
                cout = outchannel_sizes[in_layer_id]
                out_mask = hard_vector[in_layer_id]

                
                in_replacer = nn.InstanceNorm3d(num_features=cout, eps=modules[k].eps, momentum=modules[k].momentum, affine=modules[k].affine,
                                                track_running_stats=modules[k].track_running_stats)

                if new_model is None:
                    modules[k] = in_replacer
                else:
                    new_modules.append(in_replacer)

        return layer_id
    

    def _replace_model(modules, layer_id):
        keys = list(modules.keys())
        for i, k in enumerate(keys):
            if modules[k] == None:
                continue
            if len(modules[k]._modules) > 0:
                layer_id = _replace_model(modules[k]._modules, layer_id)
            if isinstance(modules[k], nn.Conv3d):
                modules[k] = new_modules[layer_id]
                layer_id += 1
            elif isinstance(modules[k], nn.BatchNorm3d):
                modules[k] = new_modules[layer_id]
                layer_id += 1
            elif isinstance(modules[k], nn.InstanceNorm3d):
                modules[k] = new_modules[layer_id]
                layer_id += 1
        return layer_id
    
    with torch.no_grad():
        _melt(model._modules, 0)
        if new_model is not None:
            _replace_model(new_model._modules, 0)


######## for kd loss ###########

def get_adaptive_layer(num_channels):
    """
        baseline feature map과 sub-network의 feature map의 channel수를 맞춰주는 1x1 conv layer list 생성
    """
    adaptive_layers = []
    for ch in num_channels:
        layer = nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=1)
        init_weights(layer, init_type='kaiming')
        adaptive_layers.append(layer)
    
    return nn.ModuleList(adaptive_layers)


def get_kd_loss(loss_name, kd_w, len_losses):
    """
        kd loss term 얻기

        Returns:
            loss           : kd loss term
            do_norm        : layer normalization을 수행하는지

    """
    if loss_name == 'Chatt':
        loss = chatt_loss(kd_w=kd_w, len_losses=len_losses)
        do_norm = True
    elif loss_name == 'MKD':
        loss = mkd_loss(kd_w=kd_w, len_losses=len_losses)
        do_norm = False
    else:
        raise Exception(f"KD Loss function {loss_name} can't be found.")
    
    return loss, do_norm


class chatt_loss(nn.Module):
    """
        channel attention loss
    """
    def __init__(self, kd_w, len_losses):
        super().__init__()
        self.kd_w = kd_w
        self.loss_func = torch.nn.MSELoss()
        self.len_losses = len_losses
    
    def forward(self, s_fmap, t_fmap, gt, adaptive_layers):
        losses = []

        for i in range(self.len_losses):
            losses.append(self.loss_func(self.get_channel_att(adaptive_layers[i](s_fmap[i])), self.get_channel_att(t_fmap[i].detach())) / (self.len_losses - i))

        return sum(losses) * self.kd_w


    def get_channel_att(self, features):
        avg_att = features.mean(dim=(2, 3, 4))
        max_att = features.amax(dim=(2, 3, 4))

        return torch.cat((avg_att, max_att), dim=1)
    

    def set_adap_kd_loss(self, adaptive_layers):
        for i, layer in enumerate(adaptive_layers):
            layer.weight.grad = layer.weight.grad * (self.len_losses - i)
            if getattr(layer, 'bias') is not None:
                layer.bias.grad = layer.bias.grad * (self.len_losses - i)


class mkd_loss(nn.Module):
    """
        masked kd loss
    """
    def __init__(self, kd_w, len_losses):
        super().__init__()
        self.kd_w = kd_w
        self.loss_func = torch.nn.MSELoss()
        self.len_losses = len_losses
    
    def forward(self, s_fmap, t_fmap, gt, adaptive_layers):
        losses = []

        # Attention Unet 3D에 맞는 mask 생성
        masks = [gt.detach()]
        for i in range(int(self.len_losses / 2)):
            masks.append(F.interpolate(masks[i], size=t_fmap[i+1].shape[2:], mode='trilinear'))
        masks = masks + [masks[-2], masks[-3], masks[-4], masks[-5]]

        for i in range(self.len_losses):
            w = masks[i].expand_as(t_fmap[i])
            m = (w != 0.)
            losses.append(self.loss_func(adaptive_layers[i](s_fmap[i])[m], t_fmap[i].detach()[m]))

        return sum(losses) * self.kd_w
    
    def set_adap_kd_loss(self, adaptive_layers):
        for i, layer in enumerate(adaptive_layers):
            layer.weight.grad = layer.weight.grad * (self.len_losses - i)
            if getattr(layer, 'bias') is not None:
                layer.bias.grad = layer.bias.grad * (self.len_losses - i)


    

    
