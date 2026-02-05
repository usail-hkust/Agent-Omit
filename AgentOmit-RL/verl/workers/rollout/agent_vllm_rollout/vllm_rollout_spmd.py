# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import json
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
# from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs
import torch.distributed as dist
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List
from omegaconf import DictConfig
import torch
import torch.distributed
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm
import time
from verl import DataProto
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams
from vllm.distributed import initialize_model_parallel
import os
import json
import time
import requests
from copy import deepcopy
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.utils.agentgym.client import init_env_client
from verl.workers.rollout.schemas import RolloutHandler, Message, _pre_process_inputs
# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics
import openai
import json
from typing import List, Dict, Optional
import requests
from openai import OpenAI
import httpx
import re


def parse_tool_response_skip(text: str) -> List[int]:
    """
    解析文本以提取所有 <omit_tool_response_N> 中的 N 值。
    
    Args:
        text (str): 包含标签的输入文本。
        
    Returns:
        List[int]: 需要跳过的步骤数列表。如果未找到标签，则返回空列表。
    """
    if not text:
        return []

    # 定义正则表达式模式
    # <omit_tool_response_  : 匹配固定的标签前缀
    # (\d+)                 : 捕获组，匹配一个或多个数字 (这就是 N)
    # >                     : 匹配标签的开始结束括号
    # </omit_tool_response_ : 匹配闭合标签前缀
    # \d+                   : 匹配闭合标签中的数字
    # >                     : 匹配闭合标签的结束括号
    pattern = r"<omit_tool_response_(\d+)></omit_tool_response_\d+>"
    
    # 在文本中查找所有匹配项
    matches = re.findall(pattern, text)
    
    # 将匹配到的字符串转换为整数列表
    result = []
    for match in matches:
        try:
            result.append(int(match))
        except ValueError:
            # 如果转换失败，跳过该项
            continue
    
    return result

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


class vLLMRollout(BaseRollout):

    def __init__(self, model_path: str, config: DictConfig, agentgym_config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.validation_mode = False
        self.rollout_mode = False
        self.model_path = model_path
        self.config = config
        self.agentgym_config = agentgym_config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens')

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
                train_tp = kwargs.get('train_tp', None)
                num_tp_per_train_tp = train_tp // tensor_parallel_size
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        #assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
        #    "model context length should be greater than total sequence length"

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError('Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill')

        trust_remote_code = kwargs.get('trust_remote_code', False)
        load_format = 'dummy' if config.load_format.startswith('dummy') else config.load_format

        limit_mm_per_prompt = None
        if config.get('limit_images', None):  # support for multi-image data
            limit_mm_per_prompt = {"image": config.get('limit_images')}
            
        enable_fp8_infer_origin_train = config.get("fp8_infer_origin_train", False)
        if enable_fp8_infer_origin_train:
            from vllm.platforms import current_platform

            assert current_platform.has_device_capability(89), (
                f"FP8 inference and origin train is just support device_capability>=sm89. Used: {current_platform.get_device_capability()}"
            )

            from .monkey_patch import monkey_patch_for_vllm_fp8

            monkey_patch_for_vllm_fp8()
            print("Add monkey patch for vllm to reload fp8 weight.")
            
        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=True,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            limit_mm_per_prompt=limit_mm_per_prompt,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            rope_scaling={"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768},
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=True,
            trust_remote_code=trust_remote_code,
            seed=config.get('seed', 0),
            # quantization='fp8' if enable_fp8_infer_origin_train else None,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            # stop=['</tool_call>', "</answer>", "</search>"],  # 
            # include_stop_str_in_output=True,  # 
        )

        # # we may detokenize the result all together later
        if vllm_version != '0.3.1':
            kwargs['detokenize'] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    def preprocess_prompt_to_rollout_handler(self, prompts: DataProto, n: int) -> List[RolloutHandler]:
        assert "raw_prompt" in prompts.non_tensor_batch.keys(), "raw_prompt is not in non_tensor_batch, need to set data.return_raw_chat=True"
        handler_list = []
        # prompt_list = prompts.non_tensor_batch["raw_prompt"]
        # # print(prompts.non_tensor_batch["raw_prompt"])
        # print(len(prompts.non_tensor_batch["raw_prompt"]))
        for i, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for _ in range(n):
                # only keep not pad part
                input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch['input_ids'][i])
                attention_mask = _pre_process_inputs(0, prompts.batch['attention_mask'][i])
                position_ids = compute_position_id_with_mask(torch.tensor(attention_mask)).tolist()
                handler = RolloutHandler(
                    messages=[
                        Message(role=prompt["role"], content=prompt["content"]) for prompt in raw_prompt
                    ],
                    task_name=prompts.non_tensor_batch["item_id"][i].split("_")[0],
                    item_id=int(prompts.non_tensor_batch["item_id"][i].split("_")[-1]),
                    score=0,
                    done=False,
                    input_ids=list(input_ids),
                    prompt_ids=list(input_ids),
                    response_ids=[],
                    attention_mask=list(attention_mask),
                    prompt_attention_mask=list(attention_mask),
                    response_attention_mask=[],
                    position_ids=list(position_ids),
                    prompt_position_ids=list(position_ids),
                    response_position_ids=[],
                    loss_mask=[0] * len(input_ids),
                    prompt_loss_mask=[0] * len(input_ids),
                    response_loss_mask=[],
                    max_response_len=self.config.response_length,
                    max_model_len=min(self.config.max_model_len, self.config.prompt_length + self.config.response_length)
                )
                assert len(handler.input_ids) == len(handler.attention_mask) == len(handler.position_ids) == len(handler.loss_mask), f"RolloutHandler has mismatched length: input_ids={len(handler.input_ids)}, attention_mask={len(handler.attention_mask)}, position_ids={len(handler.position_ids)}, loss_mask={len(handler.loss_mask)}"
                handler_list.append(handler)
        return handler_list

    
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        # if self.config.free_cache_engine:
        #     self.inference_engine.init_cache_engine()
        if hasattr(self.inference_engine, 'init_cache_engine'):
            self.inference_engine.init_cache_engine()
        else:
            # 使用新的 API 如 reset_cache 或其他方法
            pass
            
        global_steps = prompts.meta_info.get('global_steps', None)
        max_rounds = prompts.meta_info.get('max_rounds', 10)
        cur_device = prompts.batch["input_ids"].device

        do_sample = prompts.meta_info.get('do_sample', True)
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }
        if self.validation_mode:
            num_rollout = 1
        else:
            num_rollout = self.config.n
        # print('validation mode:', self.validation_mode)
        # print('num_rollout:', num_rollout)
        # print('==================')
        # repeat for self.config.n times to rollout
        batch_size = prompts.batch['input_ids'].size(0)
        batch_size *= num_rollout
        rollout_handler_ls = self.preprocess_prompt_to_rollout_handler(prompts, n=num_rollout)
        env_clients = [init_env_client(self.agentgym_config) for _ in range(batch_size)]
        time.sleep(self.config.send_interval) # take a break before sendng request
        all_done_flag = False
        for idx, rollout_handler in enumerate(rollout_handler_ls):
            try:
                env_clients[idx].reset(rollout_handler.item_id)
                if self.rollout_mode:
                    pass
                else:
                    task = env_clients[idx].observe()
                    rollout_handler.add_user_message(self.tokenizer, task)
            except TimeoutError:
                print(f"Reset Timeout: Webarena Env Timeout. item id = {rollout_handler.item_id}")
                rollout_handler.done = True
                rollout_handler.score = 0

        rounds = 0
        task_rounds = [0] * batch_size
        rollout_bar = tqdm(total = max_rounds, desc="Running rounds", disable=torch.distributed.get_rank() != 0)

        def agent_step(i, idx):
            content = self.tokenizer.decode(response_ids[i], skip_special_tokens=True)

            rollout_handler_ls[idx].add_assistant_message(self.tokenizer, content)
            task_rounds[idx] += 1
            try:
                handler = rollout_handler_ls[idx]
                user_msg_count = sum(1 for msg in handler.messages if msg.role == "user")
                token_length_count = sum(len(msg.content) for msg in handler.messages)
                omit_tool_response_num = parse_tool_response_skip(content)

                saved_token_length_count_observation = 0
                for skip_turn in omit_tool_response_num:
                    if isinstance(skip_turn, int) and skip_turn > 0:
                        target_tag = f"<tool_response_{skip_turn}>"
                        for msg in handler.messages:
                            if msg.role == "user" and target_tag in msg.content:
                                saved_token_length_count_observation += len(msg.content)
                messages__save = [msg for msg in handler.messages]
                saved_token_length_count_think = sum(len(msg.content) for msg in messages__save) / len(messages__save) if messages__save else 0

                action_payload = {
                    "action_content": content,  # 原来的 action/content
                    "parameters": {             # 你的参数字典
                        'user_msg_count': user_msg_count,
                        'saved_token_length_count_observation': saved_token_length_count_observation,
                        'saved_token_length_count_think': saved_token_length_count_think,
                        'token_length_count': token_length_count
                    }
                }
                action_str = json.dumps(action_payload)
                step_output = env_clients[idx].step(action_str, user_msg_count)
                skip_turn_list = getattr(step_output, "skip_turn", [])

                for skip_turn in skip_turn_list:
                    if isinstance(skip_turn, int) and skip_turn > 0:

                        target_tag = f"<tool_response_{skip_turn}>"
                        
                        for msg in handler.messages:
  
                            if msg.role == "user" and target_tag in msg.content:
                                # omit tool response
                                new_content = f"<omitted_tool_response_{skip_turn}></omitted_tool_response_{skip_turn}>"
                                msg.content = new_content

                                break

                state, rollout_handler_ls[idx].score, rollout_handler_ls[idx].done = (
                    step_output.state,
                    step_output.reward,
                    step_output.done,
                )
                rollout_handler_ls[idx].add_user_message(self.tokenizer, state)
                return step_output.done
            except Exception as e:
                rollout_handler_ls[idx].score = 0
                rollout_handler_ls[idx].done = True
                print(f"Rollou step Error: {e} item id = {rollout_handler_ls[idx].item_id}")
                return True
        
        while rounds < max_rounds and not all_done_flag:
            # get generation prompt
            generation_prompt_idxs = []
            generation_messgaes_idxs = []
            # rollout_deepseek_api_id = []
            not_done_idxs = []
            for idx, rollout_handler in enumerate(rollout_handler_ls):
                if not rollout_handler.done:
                    # rollout_deepseek_api_id.append(rollout_deepseek_api[idx])
                    generation_prompt_idxs.append(rollout_handler.get_generation_prompt(self.tokenizer))
                    generation_messgaes_idxs.append(rollout_handler.messages)
                    not_done_idxs.append(idx)

            rollout_bar.set_description(f"Rounds {rounds + 1}/{max_rounds} | Active agents per gpu: {len(not_done_idxs)}")
            # users can customize different sampling_params at different run

            if vllm_version not in ('0.4.2', '0.5.4', '0.6.3'):
                kwargs = {
                    'best_of': 1,
                    'top_p': 1.0,
                    'top_k': -1,
                    'min_p': 0.0,
                    'temperature': 0,
                    'n': 1,  # if greedy, only 1 response
                    'stop': ["</tool_call>", "</answer>"],
                    'include_stop_str_in_output': True,
                    'detokenize': True
                }
            with self.update_sampling_params(**kwargs):
                # print(self.sampling_params)
                # decoded_text = self.tokenizer.decode(
                #     generation_prompt_idxs[0], 
                #     skip_special_tokens=True  # 通常设为 True 来去除特殊标记
                # )
                # print(decoded_text)
                output_list = self.inference_engine.generate(
                    prompts=None,
                    prompt_token_ids=generation_prompt_idxs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False)
            if vllm_version in ('0.4.2', '0.5.4', '0.6.3'): 
                response_ids = output_list[0].tolist()
            else:
                response_ids = []
                for output in output_list:
                    for sample_id in range(len(output.outputs)):
                        response_ids.append(output.outputs[sample_id].token_ids)
                # response_ids = output[0].outputs[0].token_ids
            
            
            all_done_flag = True
            time.sleep(self.config.send_interval) # take a break before sendng request
            if len(not_done_idxs) > 0:
                with ThreadPoolExecutor(max_workers=len(not_done_idxs)) as executor:
                    step_dones = list(executor.map(
                        lambda args: agent_step(*args), [(i, idx) for i, idx in enumerate(not_done_idxs)]
                    ))
                    all_done_flag = all(step_dones)
            if self.rollout_mode:
                handler_ = rollout_handler_ls[idx]
                user_msg_count = sum(1 for msg in handler_.messages if msg.role == "user")
                rounds = user_msg_count-1
            else:
                rounds += 1
            rollout_bar.update(1)
        
        # process ids
        rollout_bar.close()
        response_ids, response_attention_mask, response_position_ids, response_loss_mask = [], [], [], []
        scores, messages = [], []

        for rollout_handler in rollout_handler_ls:
            # check length
            rollout_handler.truncate_output_ids()
            assert len(rollout_handler.input_ids) == len(rollout_handler.attention_mask) == len(rollout_handler.position_ids) == len(rollout_handler.loss_mask), f"""Rollout Handler has different length of {len(rollout_handler.input_ids)=}, 
            {len(rollout_handler.attention_mask)=}, {len(rollout_handler.position_ids)=}, {len(rollout_handler.loss_mask)=}"""
            assert len(rollout_handler.input_ids) <= self.config.max_model_len, f"Rollout Handler has sequence length {len(rollout_handler.input_ids)} > max_sequence_length {self.config.max_model_len}"

            response_ids.append(torch.tensor(rollout_handler.response_ids, dtype=torch.int, device=cur_device))
            response_attention_mask.append(torch.tensor(rollout_handler.response_attention_mask, dtype=torch.int, device=cur_device))
            response_position_ids.append(torch.tensor(rollout_handler.response_position_ids, dtype=torch.int, device=cur_device))
            response_loss_mask.append(torch.tensor(rollout_handler.response_loss_mask, dtype=torch.int, device=cur_device))
            scores.append(rollout_handler.score)
            messages.append(rollout_handler.messages)
        
        # pad to length
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)
        response_length = response_ids.size(1)
        delta_position_ids = torch.arange(1, response_length + 1, device=cur_device)
        delta_position_ids = delta_position_ids.unsqueeze(0).repeat(batch_size, 1)
        input_ids = prompts.batch['input_ids']  # (bs, prompt_length)
        prompt_length = input_ids.size(-1)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']
        input_ids = input_ids.repeat_interleave(num_rollout, dim=0)
        attention_mask = attention_mask.repeat_interleave(num_rollout, dim=0)
        position_ids = position_ids.repeat_interleave(num_rollout, dim=0)
        response_position_ids = position_ids[:, -1:] + delta_position_ids

        seq = torch.cat((input_ids, response_ids), dim=-1)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((position_ids, response_position_ids), dim=-1)
        response_mask = response_loss_mask

        reward_tensor = torch.zeros_like(response_ids, dtype=torch.float32) # (bs, response_length)
        valid_response_length = attention_mask[:, prompt_length:].sum(dim=-1)
        for i in range(len(scores)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]

        if global_steps:
            try:
                os.makedirs(os.path.join(self.config.rollout_log_dir, f"step{global_steps}"), exist_ok=True)
                with open(os.path.join(self.config.rollout_log_dir, f"step{global_steps}/{torch.distributed.get_rank()}.json"), "w") as f:
                    json_msg = []
                    for idx, msgs in enumerate(messages):
                        records = {
                            "item_id": str(rollout_handler_ls[idx].task_name) + '_' + str(rollout_handler_ls[idx].item_id),
                            "conversations": [msg.to_dict() for msg in msgs],
                            "reward": scores[idx]
                        }
                        json_msg.append(records)
                    json.dump(json_msg, f, ensure_ascii=True, indent=4)
            except Exception as e:
                print(e)

        # close clients
        for client in env_clients:
            try:
                client.close()
            except Exception as e:
                print(f"Error during closing env: {e}")

        batch = TensorDict(
            {
                'prompts': input_ids,
                'responses': response_ids,
                'input_ids': seq,
                'attention_mask': attention_mask,
                'position_ids': position_ids,
                'response_mask': response_mask,
                'scores': reward_tensor,
                'task_rounds': torch.tensor(task_rounds, dtype=torch.float32).to(input_ids.device),
                'task_scores': reward_tensor
            },
            batch_size=batch_size)
        
        # free vllm cache engine
        if self.config.free_cache_engine:
            self.free_vllm_cache()

        return DataProto(batch=batch)



    def free_vllm_cache(self):
        """释放 vLLM 缓存的安全方法"""
        if not hasattr(self, 'inference_engine'):
            return
            
        try:
            # 方法1: 通过调度器清理
            if hasattr(self.inference_engine, 'llm_engine'):
                scheduler = self.inference_engine.llm_engine.scheduler
                if hasattr(scheduler, 'block_manager'):
                    scheduler.block_manager.clear()
            
            # 方法2: 重置模型状态
            if hasattr(self.inference_engine, 'model_executor'):
                self.inference_engine.model_executor = None
                
            # 方法3: 强制垃圾回收
            import torch
            import gc
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print("vLLM 缓存已清理")
            
        except Exception as e:
            print(f"清理 vLLM 缓存失败: {e}")