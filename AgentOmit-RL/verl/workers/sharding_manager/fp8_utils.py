import torch
from vllm import _custom_ops as ops


# moe support?
stacked_params_mapping = [
    # (param_name, shard_name, shard_id)
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]


def block_scale_and_split_gen(
    params_bf16, name_list, do_full_tensor, merge_dim=0, split_dim=1
):
    param_list = [
        params_bf16[name].full_tensor() if do_full_tensor else params_bf16[name]
        for name in name_list
    ]
    merged_param = torch.cat(param_list, dim=merge_dim)

    ref_y, inv_scale = ops.scaled_fp8_quant(merged_param, None)
    split_param = torch.split(
        ref_y.transpose(0, 1), [p.shape[0] for p in param_list], dim=split_dim
    )

    for name, split_param in zip(name_list, split_param):
        yield (name, split_param)
        yield (name + "_scale", inv_scale)


def quant_weight_to_fp8_gen(fp8_model, full_params_bf16, do_full_tensor=False):
    scale_layer_name_list = {
        name[:-6] for name, _ in fp8_model.named_parameters() if ".weight_scale" in name
    }

    for name, param in full_params_bf16.items():
        vllm_name = name
        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name in name:
                vllm_name = name.replace(weight_name, param_name)
                break

        if vllm_name in scale_layer_name_list:
            if "q_proj" in name or "k_proj" in name or "gate_proj" in name:
                continue

            if "v_proj" in name:
                q_proj_name = name.replace("v_proj", "q_proj")
                k_proj_name = name.replace("v_proj", "k_proj")
                yield from block_scale_and_split_gen(
                    full_params_bf16, [q_proj_name, k_proj_name, name], do_full_tensor
                )
            elif "up_proj" in name:
                gate_proj_name = name.replace("up_proj", "gate_proj")
                yield from block_scale_and_split_gen(
                    full_params_bf16, [gate_proj_name, name], do_full_tensor
                )
            else:
                ref_y, inv_scale = ops.scaled_fp8_quant(
                    param.full_tensor() if do_full_tensor else param, None
                )
                yield (name, ref_y.t())
                yield (name + "_scale", inv_scale)
        else:
            yield (name, param.full_tensor() if do_full_tensor else param)
