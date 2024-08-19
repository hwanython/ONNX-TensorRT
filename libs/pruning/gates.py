import torch.nn as nn
import torch.nn.functional as F
import torch
import uuid
import numpy as np

import random


class virtual_gate(nn.Module):
    """
        모델에 삽입되는 Gates
        모델 선언시 gate_flag arguments에 의해 하나의 Convolution layer뒤에 연결
        forwarding시 Hyper network(HyperNet)에서 gate들의 출력값(gate_f)을 전달 받아 Convolution layer에서 생성되는 feature map과 연산 

        class varialbles:
            width           : layer안 filter 수
            gate_f          : gate들의 출력 값 (soft or hard)
            group_id        : group (skip connection을 위한 grouping) id
    """
    
    def __init__(self, width, group_id=None, prune=True):
        super().__init__()
        self.width = width
        self.prune = prune
        self.gate_f = torch.ones(width)
        if group_id == None:
            self.group_id = uuid.uuid1()
        else:
            self.group_id = group_id
        
    def extra_repr(self):
        return ('width={width}, g_id={group_id}, is_prune={prune}'.format(**self.__dict__))
    
    def forward(self, input):
        """
        forwarding시 convolution layer의 출력 feature map은 gate의 출력값들(gate_f)과 곱셈 연산을 수행
        if gate 출력값 = 0  --> 다음 layer에 입력되는 해당 feature map channel은 0 (filter가 off인것과 동일)
        """
        if len(input.size()) == 2:
            if input.is_cuda:
                gate_f = self.gate_f.to(input.device)
            input = gate_f.expand_as(input) * input
            
            return input
        
        elif len(input.size()) == 4:
            gate_f = self.gate_f.unsqueeze(-1).unsqueeze(-1)
            
            if input.is_cuda:
                gate_f = gate_f.to(input.device)    
            input = gate_f.expand_as(input) * input
            
            return input
        
        elif len(input.size()) == 5:
            gate_f = self.gate_f.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            if input.is_cuda:
                gate_f = gate_f.to(input.device)    
            input = gate_f.expand_as(input) * input
            
            return input
        
    def set_structure_value(self, value):
        """
        Hyper network에서 gate의 출력 값을 전달받기
        """
        self.gate_f = value


class simple_gate(nn.Module):
    """
        하나의 Convolution layer안에 존재하는 filter들의 on / off 여부를 학습하는 gate
        Group(skip connection에 의해 묶이는 layer)당 하나의 simple gate 생성
        Decomposed differentiable gate를 위해 Scale 및 Mask로 분류

        Parameters:
            scale           : gate의 최적의 magnitude를 학습
            mask            : filter의 제거 여부를 학습
            width           : layer안 filter 수
    """
    def __init__(self, width):
        super(simple_gate, self).__init__()
        self.mask = nn.Parameter(torch.zeros(width))
        self.scale = nn.Parameter(torch.zeros(width))
        self.width = width
 
    def forward(self):
        return self.scale, self.mask


class HyperNet(nn.Module):
    """
        모델 안 존재하는 모든 gates들을 통제하는 network
        Paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Gao_Network_Pruning_via_Performance_Maximization_CVPR_2021_paper.pdf
        Code: https://github.com/gaosh/NPPM

        class variables:
            structure           : layer별 filter 수
            group_ids           : 모델안 모든 group들의 id 집합
            T                   : sigmoid의 temperature (gate의 sharpness를 조절)
            base                : sigmoid의 offset
            device              : cpu or gpu
    """

    def __init__(self, structure, group_ids, no_prune_idx, T=0.4, base = 3, device='cuda'):
        super(HyperNet, self).__init__()
        self.structure = structure
        self.group_ids = group_ids
        self.no_prune_idx = no_prune_idx
        self.T = T
        self.base = base
        self.device = device
        self.get_module_list()
        
    def forward(self, sample=True):
        """ 
            gate들의 출력
            
            Returns:
                soft_vector         : layer별 학습을 위한 sigmoid출력값 (0 ~ 1 사이의 continuous한 값)
                hard_vector         : layer별 hard 출력값 (0 or 1). 각 필터의 on / off여부를 나타냄
                arch_vector         : 각 candidate들에 의해 형성되는 가상의 pruned network의 layer당 filter 수
        """
        soft_gate, hard_gate = [], []
        
        for i, p in enumerate(self.p_list):
            soft, hard = self.get_soft_hard(p=p, sample=sample)
            soft_gate.append(soft)
            hard_gate.append(hard)

        soft_vector, hard_vector, arch_vector =  self.group_indexing(soft_gate=soft_gate, hard_gate=hard_gate)
        
        return soft_vector, hard_vector, arch_vector

    def get_module_list(self):
        """ 
            group 별로 gates들을 선언
            동일한 group안의 layer들을 하나의 gate만을 선언(하나의 simple gate가 group안 모든 layer속 filter들을 통제)

            class variables:
                group_count         : 하나의 group에 몇개의 layer들이 포함되어 있는지
                p_list              : 모델안 전체 gates(group)들의 집합 (group당 하나의 simple_gate를 선언)
                idx_list            : convolution layer들이 어느 gates(group)에 포함되는지 (p_list의 어느 simple gate에 해당되는지)
        """
        module_list = []
        g_ids = []
        group_count = []
        idx_list = []
        group_info = {}
        for i in range(len(self.structure)):
            if i in self.no_prune_idx:
                idx_list.append(-1)
            elif self.group_ids[i] in group_info:
                idx_list.append(group_info[self.group_ids[i]]['idx'])
                group_info[self.group_ids[i]]['count'] += 1
                
            else:
                group_info[self.group_ids[i]] = {
                    'count': 1,
                    'idx': len(module_list)
                }
                module_list.append(simple_gate(self.structure[i]))
                idx_list.append(group_info[self.group_ids[i]]['idx'])
                g_ids.append(self.group_ids[i])
                
        for g in g_ids:
            group_count.append(group_info[g]['count'])
        
        self.group_count = group_count
        self.idx_list = idx_list
        self.p_list = nn.ModuleList(module_list)
    
    def set_group_grad(self):
        """
        group별 gradient scaling 
        """
        for i in range(len(self.p_list)):
            self.p_list[i].mask.grad = self.p_list[i].mask.grad / self.group_count[i]
            self.p_list[i].scale.grad = self.p_list[i].scale.grad / self.group_count[i]


    def get_soft_hard(self, p, sample=True):
        """
            gates들의 soft, hard 값 출력
            
            Args:
                p               : gates들의 weight (scale, mask)
                sample          : gaussian 분포에서 sampling된 value를 활용할지
            
            Returns:
                soft            : 하나의 group을 관리하는 gate(simple gate)의 sigmoid출력값 (0 ~ 1 사이의 continuous한 값)
                hard            : 하나의 group을 관리하는 gate(simple gate)의 hard 출력값 (0 or 1). 각 필터의 on / off여부를 나타냄
        """
        
        s, m = p()
        mask = m + self.base
        scale = s + self.base

        if sample:
            gaussian_sample = torch.randn(m.size())
            if m.is_cuda:
                gaussian_sample = gaussian_sample.cuda()
            mask = mask + gaussian_sample
        
        mask = torch.sigmoid(mask / self.T)
        scale = torch.sigmoid(scale / self.T)
        
        hard = torch.zeros(mask.size()).to(mask.device)
        hard[mask>=0.5] = 1.
        hard[mask<0.5] = 0.
        hard = torch.logical_or(hard, F.one_hot(torch.argmax(mask), num_classes=m.size(-1))).float()
        hard = (hard - mask).detach() + mask
        soft = scale * hard
        
        return soft, hard
    
    def group_indexing(self, soft_gate, hard_gate):
        """
            p_list에 group 별 gate 출력값들을 모델안 Convolution layer들에 matching

            Args:
                soft_gate           : group 별 gate의 soft출력값 (0 ~ 1 사이의 continuous한 값)
                hard_gate           : group 별 gate의 hard출력값 (0 or 1). 각 필터의 on / off여부를 나타냄
            
            Returns:
                soft_vector         : layer별 학습을 위한 sigmoid출력값 (0 ~ 1 사이의 continuous한 값)
                hard_vector         : layer별 hard 출력값 (0 or 1). 각 필터의 on / off여부를 나타냄
                arch_vector         : 각 candidate들에 의해 형성되는 가상의 pruned network의 layer당 filter 수
        """
        soft_vector, hard_vector, arch_vector = [], [], []

        for i, idx in enumerate(self.idx_list):
            if idx == -1:
                soft_vector.append(torch.ones(self.structure[i]).to(self.device))
                hard_vector.append(torch.ones(self.structure[i]))
                arch_vector.append(torch.tensor(self.structure[i]).to(self.device))
            else:
                soft_vector.append(soft_gate[idx])
                hard_vector.append(hard_gate[idx].detach().cpu())
                arch_vector.append(hard_gate[idx].sum())
        return soft_vector, hard_vector, arch_vector