"""Devices.

Utilities to get the correct device, and assist in distributing model layers across multiple
devices.
"""
from __future__ import annotations

import torch
import logging
import transformer_lens
from torch import nn
from typing import Optional, Union, Dict
from collections import OrderedDict


logging.basicConfig( level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)

def multi_device_setup(device: Union[str, torch.device, Dict[str, str]]):
    multi_device = False
    if isinstance(device, Dict[str, str]) and len(device)>1:
        multi_device = True
    return multi_device

def get_sorted_available_gpus(devices : Optional[List[str]] = None):
    if devices is None:
        devices = []
        for i in range(torch.cuda.device_count()):
            device_name = f'cuda:{i}'
            free, total = torch.cuda.mem_get_info(device=device_name)
            devices.append((device_name, free))
    else:
        for i,d in enumerate(devices):
            device_name = d
            free, total = torch.cuda.mem_get_info(device=device_name)
            devices[i] = (d, free)
    return sorted(devices, key=lambda x: x[1], reverse=True)

def estimate_model_size(cfg: HookedTransformerConfig):
    overhead_multiplier = 1.2

    bytes_by_dtype = {
        torch.uint8: 1,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float32: 4
    }
    bytes = bytes_by_dtype[cfg['dtype']]
    embed_layer_p_num = cfg['d_vocab']*cfg['d_model'] # [d_vocab, d_model]

    w_q_k_v_p_num = cfg['d_model']*cfg['n_heads']*cfg['d_head']
    b_q_k_v_p_num = cfg['n_heads']*cfg['d_head']
    w_o_p_num = cfg['n_heads']*cfg['d_head']*cfg['d_model']
    b_o_p_num = cfg['d_model']
    w_gate_p_num = cfg['d_model']*cfg['d_mlp']
    w_in_p_num = cfg['d_model']*cfg['d_mlp']
    b_in_p_num = cfg['d_mlp']
    w_out_p_num = cfg['d_mlp']*cfg['d_model']
    b_out_p_num = cfg['d_model']
    single_transformer_block_p_num = ((w_q_k_v_p_num*3) + (b_q_k_v_p_num*3) + w_o_p_num + b_o_p_num + w_gate_p_num + w_in_p_num + b_in_p_num + w_out_p_num + b_out_p_num)
    total_transformer_blocks_p_num = cfg['n_layers']*single_transformer_block_p_num

    unembed_p_num = cfg['d_vocab']*cfg['d_model'] # [d_model, d_vocab]
    ln_final_p_num = cfg['d_model']
    b_unembed_p_num = cfg['d_vocab']

    total_p_num = embed_layer_p_num + total_transformer_blocks_p_num + unembed_p_num + ln_final_p_num + b_unembed_p_num
    total_size = total_p_num*bytes*overhead_multiplier
    module_map_with_size = OrderedDict([
        ('embed', embed_layer_p_num*bytes*1.2)])
    for l in range(cfg['n_layers']):
        module_map_with_size.update(
            {f'blocks.{l}': single_transformer_block_p_num*bytes*overhead_multiplier}
        )
    module_map_with_size.update({
        'ln_final': ln_final_p_num,
        'unembed': (unembed_p_num + b_unembed_p_num)*bytes*overhead_multiplier})
    return total_size, module_map_with_size

def expand_device_map(device_map: Union[str, torch.device, Dict[str, str]], cfg: Dict):
    """ For now, we won't split the components of a layer across GPUs, but just split whole layers. """
    blocks = []
    expanded_device_map = OrderedDict()
    blocks.append('embed')
    blocks.extend([f'blocks.{i}' for i in range(cfg['n_layers'])])
    blocks.append('ln_final')
    blocks.append('unembed')
    devices = []
    if isinstance(device_map, str):
        if device_map in ['auto', 'balanced_low_0']:
            devices = get_sorted_available_gpus()
        elif 'cuda' in device_map or 'cpu' in device_map:
            devices = get_sorted_available_gpus([device_map])
    elif isinstance(device_map, Dict):
        for device_id, device_name in device_map.items():
            devices.append(device_name)
        devices = get_sorted_available_gpus(devices)

    if len(devices)>1 and 'cpu' in devices:
        if not torch.cuda.is_available():
            logging.warning('CUDA is not available on the machine. Since the device_map includes the CPU, we will load the model entirely into this device.')
            devices = 'cpu'
    logging.info(f"Available devices with their memory: {devices}")

    estimated_model_size, module_map_with_size = estimate_model_size(cfg)

    cuda_devices = [d for d in devices if 'cuda' in d[0]]
    mem_allocated_on_current_device = 0
    if device_map == 'auto':
        device_memory_threshold = estimated_model_size/len(cuda_devices)
        device_cursor = 0
        for b in blocks:
            if mem_allocated_on_current_device <= device_memory_threshold:
                mem_allocated_on_current_device += module_map_with_size[b]
            else:
                mem_allocated_on_current_device = module_map_with_size[b]
                device_cursor = device_cursor + 1
            expanded_device_map[b] = cuda_devices[device_cursor][0]
    elif device_map == 'balanced_low_0':
        device_cursor = len(cuda_devices) - 1
        for b in range(blocks, 0, -1):
            current_device_memory = cuda_devices[device_cursor][1]
            if mem_allocated_on_current_device < current_device_memory:
                expanded_device_map[b] = cuda_devices[device_cursor][0]
                mem_allocated_on_current_device += module_map_with_size[b]
            else:
                mem_allocated_on_current_device = 0
                device_cursor = device_cursor -1
        # In case we didn't get to put something inside our first GPU during the split, we can actually use if completely for the data.
        if device_cursor > 0:
            expanded_device_map['embed'] = cuda_devices[0][0]
    else:
        for b in blocks:
            expanded_device_map[b] = devices[0][0]
    return expanded_device_map
