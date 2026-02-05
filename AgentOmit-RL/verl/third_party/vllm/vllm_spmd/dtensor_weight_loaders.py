# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The vLLM team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/model_loader

from typing import Dict
import torch
import torch.nn as nn
from typing import Dict, Any
import torch.nn as nn
from torch.distributed._tensor import DTensor
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import is_pp_missing_parameter


def gemma_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        for param_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            stacked_name = name.replace(shard_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if stacked_name.endswith(".bias") and stacked_name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[stacked_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # lm_head is not used in vllm as it is tied with embed_token.
            # To prevent errors, skip loading lm_head.weight.
            if "lm_head.weight" in name:
                continue
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def gptbigcode_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "lm_head.weight" in name:
            continue
        if ".attn.bias" in name:
            # Skip attention mask.
            # NOTE: "c_attn.bias" should not be skipped.
            continue
        local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
        param = params_dict[name]
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def starcoder2_dtensor_load_weights(actor_weights: Dict, vllm_model: nn.Module):
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def llama_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        (".qkv_proj", ".q_proj", "q"),
        (".qkv_proj", ".k_proj", "k"),
        (".qkv_proj", ".v_proj", "v"),
        (".gate_up_proj", ".gate_proj", 0),
        (".gate_up_proj", ".up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters())
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            # Models trained using ColossalAI may include these tensors in
            # the checkpoint. Skip them.
            continue
        # With tie_word_embeddings, we can skip lm_head.weight
        # The weight might appear unnecessarily in the files if the model is
        # processed with quantization, LoRA, fine-tuning, etc.
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight)


def qwen2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def qwen2vl_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


from vllm.model_executor.layers.fused_moe import FusedMoE


def deepseekv2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]

    # Params for weights, fp8 weight scales, fp8 activation scales
    # (param_name, weight_name, expert_id, shard_id)
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=vllm_model.config.n_routed_experts,
    )

    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    for name, loaded_weight in actor_weights.items():
        if "rotary_emb.inv_freq" in name:
            continue
        for param_name, weight_name, shard_id in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name = name.replace(weight_name, param_name)
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, vllm_model):
                continue

            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            break
        else:
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(
                    param,
                    local_loaded_weight.to(dtype=param.dtype),
                    weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                )
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, vllm_model):
                    continue

                param = params_dict[name]
                local_loaded_weight = redistribute_dtensor(param_name=name, loaded_weights=loaded_weight)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, local_loaded_weight.to(dtype=param.dtype))


def gpt2_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    pass

def qwen3_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    """
    Qwen3 DTensor weight loader
    
    根据Qwen3的模型架构特点，其参数结构与Qwen2类似[citation:1]，
    但需要注意Ollama v0.12.2对Qwen3架构的特殊支持[citation:1]
    """
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    
    for name, loaded_weight in actor_weights.items():
        # 跳过旋转嵌入参数[citation:1]
        if "rotary_emb.inv_freq" in name:
            continue
            
        # 如果词嵌入被绑定，跳过lm_head.weight
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
            
        # 处理堆叠参数
        param_loaded = False
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
                
            # 替换参数名称
            new_name = name.replace(weight_name, param_name)
            
            # 跳过GPTQ模型的额外偏置
            if new_name.endswith(".bias") and new_name not in params_dict:
                continue
                
            # 重新分布DTensor并加载权重
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            param_loaded = True
            break
            
        if not param_loaded:
            # 处理非堆叠参数
            # 跳过GPTQ模型的额外偏置
            if name.endswith(".bias") and name not in params_dict:
                continue
                
            # 跳过不存在的参数（管道并行情况下）
            if is_pp_missing_parameter(name, vllm_model):
                continue
                
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor(
                param_name=name, loaded_weights=loaded_weight
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def qwen3_dtensor_weight_loader(actor_weights: Dict, vllm_model: nn.Module) -> nn.Module:
    """
    Qwen3 DTensor weight loader
    
    根据Qwen3的模型架构特点，其参数结构与Qwen2类似[citation:1]，
    但需要注意Ollama v0.12.2对Qwen3架构的特殊支持[citation:1]
    """
    stacked_params_mapping = [
        # (param_name, shard_name, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    
    for name, loaded_weight in actor_weights.items():
        # 跳过旋转嵌入参数[citation:1]
        if "rotary_emb.inv_freq" in name:
            continue
            
        # 如果词嵌入被绑定，跳过lm_head.weight
        if vllm_model.config.tie_word_embeddings and "lm_head.weight" in name:
            continue
            
        # 处理堆叠参数
        param_loaded = False
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
                
            # 替换参数名称
            new_name = name.replace(weight_name, param_name)
            
            # 跳过GPTQ模型的额外偏置
            if new_name.endswith(".bias") and new_name not in params_dict:
                continue
                
            # 重新分布DTensor并加载权重
            local_loaded_weight = redistribute_dtensor_v2(
                param_name=name, loaded_weights=loaded_weight, vllm_model=vllm_model
            )
            param = params_dict[new_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype), shard_id)
            param_loaded = True
            break
            
        if not param_loaded:
            # 处理非堆叠参数
            # 跳过GPTQ模型的额外偏置
            if name.endswith(".bias") and name not in params_dict:
                continue
                
            # 跳过不存在的参数（管道并行情况下）
            if is_pp_missing_parameter(name, vllm_model):
                continue
                
            param = params_dict[name]
            local_loaded_weight = redistribute_dtensor_v2(
                param_name=name, loaded_weights=loaded_weight, vllm_model=vllm_model
            )
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, local_loaded_weight.to(dtype=param.dtype))

def redistribute_dtensor_v2(param_name: str, loaded_weights, vllm_model: nn.Module = None, parallelize_plan: Dict = None):
    """
    处理权重重分布，支持DTensor和普通Tensor
    """
    # 检查是否为DTensor
    if hasattr(loaded_weights, 'redistribute') and hasattr(loaded_weights, 'device_mesh'):
        # DTensor处理逻辑
        param_name = _process_parameter_names(name=param_name)
        if parallelize_plan is not None:
            assert (
                param_name in parallelize_plan.keys()
            ), f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
            placement = parallelize_plan[param_name]
            local_loaded_weights = loaded_weights.redistribute(
                device_mesh=loaded_weights.device_mesh, placements=placement
            ).to_local()
        else:
            # 如果没有并行化计划，使用full_tensor()获取完整张量
            local_loaded_weights = loaded_weights.full_tensor()
    else:
        # 普通Tensor处理逻辑 - 直接返回
        local_loaded_weights = loaded_weights
    
    return local_loaded_weights

def _process_parameter_names(name: str) -> str:
    """
    处理参数名称，移除可能的模块前缀
    """
    # 移除常见的模块前缀
    prefixes = ["model.", "transformer.", "module."]
    for prefix in prefixes:
        if name.startswith(prefix):
            name = name[len(prefix):]
    return name

def is_pp_missing_parameter(name: str, vllm_model: nn.Module) -> bool:
    """
    检查在管道并行情况下参数是否缺失
    """
    params_dict = dict(vllm_model.named_parameters(remove_duplicate=False))
    return name not in params_dict

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor, shard_id: any = None):
    """
    默认权重加载器
    """
    # 根据参数维度进行适当的权重加载
    if shard_id is not None:
        # 分片加载逻辑
        if param.dim() == 2:
            # 2D权重矩阵的分片加载
            if isinstance(shard_id, int):
                # 按行分片
                shard_size = param.shape[0] // 2
                start_idx = shard_id * shard_size
                end_idx = start_idx + shard_size
                param.data[start_idx:end_idx] = loaded_weight
            elif isinstance(shard_id, str):
                # 按列分片（用于QKV投影）
                if shard_id == "q":
                    q_size = param.shape[1] // 3
                    param.data[:, :q_size] = loaded_weight
                elif shard_id == "k":
                    q_size = param.shape[1] // 3
                    k_size = q_size
                    param.data[:, q_size:q_size+k_size] = loaded_weight
                elif shard_id == "v":
                    q_size = param.shape[1] // 3
                    k_size = q_size
                    param.data[:, q_size+k_size:] = loaded_weight
    else:
        # 完整权重加载
        param.data.copy_(loaded_weight)

def redistribute_dtensor(param_name: str, loaded_weights: DTensor, parallelize_plan: Dict = None):
    param_name = _process_parameter_names(name=param_name)
    if parallelize_plan is not None:
        assert (
            param_name
            in parallelize_plan.keys()), f"param name: {param_name} not in parallelize_plan :{parallelize_plan.keys()}"
        placement = parallelize_plan[param_name]
        local_loaded_weights = loaded_weights.redistribute(device_mesh=loaded_weights.device_mesh,
                                                           placements=placement).to_local()
    else:
        local_loaded_weights = loaded_weights.full_tensor()
    return local_loaded_weights


def _process_parameter_names(name):
    # Remove '.weight' if it exists at the end of the string
    if name.endswith(".weight"):
        name = name[:-7]

    # Remove 'model.layers.x.' or 'model.' prefix
    if "model.layers" in name:
        parts = name.split(".")
        # Reconstruct the string without 'model.layers.x.'
        name = ".".join(parts[3:])  # parts[0] is 'model', parts[1] is 'layers', parts[2] is 'x'
    elif name.startswith("model."):
        name = name[6:]  # Remove 'model.'

    return name


__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__ = {
    "GPT2LMHeadModel": gpt2_dtensor_weight_loader,
    "LlamaForCausalLM": llama_dtensor_weight_loader,
    "LLaMAForCausalLM": llama_dtensor_weight_loader,
    "MistralForCausalLM": llama_dtensor_weight_loader,  # mistral is the same as llama in vLLM
    "InternLMForCausalLM": llama_dtensor_weight_loader,
    "AquilaModel": llama_dtensor_weight_loader,
    "AquilaForCausalLM": llama_dtensor_weight_loader,
    "Phi3ForCausalLM": llama_dtensor_weight_loader,
    "GemmaForCausalLM": gemma_dtensor_weight_loader,
    "Gemma2ForCausalLM": gemma_dtensor_weight_loader,
    "GPTBigCodeForCausalLM": gptbigcode_dtensor_load_weights,
    "Starcoder2ForCausalLM": starcoder2_dtensor_load_weights,
    "Qwen2ForCausalLM": qwen2_dtensor_weight_loader,
    "Qwen3ForCausalLM": qwen3_dtensor_weight_loader,  # 新增Qwen3支持
    "DeepseekV2ForCausalLM": deepseekv2_dtensor_weight_loader,
    "Qwen2VLForConditionalGeneration": qwen2vl_dtensor_weight_loader,
}


# the actor model is .state_dict()
# Load dtensor weights
def load_dtensor_weights(actor_weights: Dict, vllm_model: nn.Module):
    weight_loader = _get_model_weight_loader(vllm_model.__class__.__name__)
    weight_loader(actor_weights, vllm_model)
    # NOTE(sgm) to reduce peak memory usage, we offload vllm model to cpu
    # after init, and we need this after sync model weights for in first iter.
    vllm_model = vllm_model.cuda()


def _get_model_weight_loader(arch: str):
    if arch in __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__:
        return __MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__[arch]
    raise ValueError(f"Model architectures {arch} are not supported for now. "
                     f"Supported architectures: {__MODEL_DTENSOR_WEIGHT_LOADER_REGISTRY__.keys()}")


# NOTE(sgm): we use per-parameter weight loader in each vllm sub
def update_dtensor_weight_loader():
    pass
