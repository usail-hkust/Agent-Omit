from typing import Any, Mapping, Dict, List, Optional
import json
import requests
from requests.exceptions import RequestException
from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


search_qa_sys_prompt = """You are answering the provided question via invoking search engine tool. If you do not have enough knowledge, issue a <tool_call>...write what you want to search hear...</tool_call> and then STOP. Do not generate <tool_response> or <answer> yet. Wait for external input wrapped in <tool_response>...external information...</tool_response>. After receiving information, reason again in <think>. If confident, output your final answer in <answer>...</answer>.

Your output must strictly follow this format:
<think>your thoughts.</think>
<tool_call>your search content.</tool_call> 
<tool_response_N>your observation after invoking search.</tool_response_N>
<think>your thoughts.</think>
...(continue to generate <tool_call>...</tool_call> for problem-solving, or generate <answer>...</answer> if the task is finished)...
<answer>...</answer>

Reminder: 
1. Do not output <answer> before receiving <tool_response> unless you are fully confident.
2. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Follow this process every time.
"""

efficient_search_qa_sys_prompt = """Answer the provided question via invoking search engine. If you do not have enough knowledge, issue a <tool_call>...write what you want to search hear...</tool_call> and then STOP. Do not generate <tool_response> or <answer> yet. Wait for external input wrapped in <tool_response>...external information...</tool_response>. After receiving information, reason again in <think>. If confident, output your final answer in <answer>...</answer>.

You use "<think>your thoughts.</think>" for in-depth thinking process; or user "<think></think>" if you think you can directly generate correct tool call action without any thinking process. 
You use "<tool_call>your next action.</tool_call>" for next action; or use "<omit_tool_response_N></omit_tool_response_N><tool_call>your next action.</tool_call>" to simultaneously generate the next action and omit prior tool responses at turn N to save context.

Your output must strictly follow this format:
<think>your thoughts.</think> (or <think></think>)
<tool_call>your search content.</tool_call> (or <omit_tool_response_N></omit_tool_response_N><tool_call>your next action.</tool_call>)
<tool_response_N>your observation after invoking search.</tool_response_N> (or <omitted_tool_response_N></omitted_tool_response_N>)
<think>your thoughts.</think> (or <think></think>)
...(continue to generate tool_call for problem-solving, or generate <answer>...</answer> if the task is finished)...
<answer>...</answer>

Reminder: 
1. Do not output <answer> before receiving <tool_response> unless you are fully confident.
2. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Follow this process every time.
3. "<think></think>" is a good way to save context when you are confident about your next action.
4. "<omit_tool_response_N></omit_tool_response_N>" can help you save context by omitting prior tool responses at turn N, you are encouraged to use when there have too many turns or are clearly stuck on a given step.

Let's start! Do remember to derive "<think> </think>" or "<omit_tool_response_N></omit_tool_response_N>" when necessary to save context!

"""


class SearchQAEnvClient(BaseEnvClient):
    conversation_start = (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": efficient_search_qa_sys_prompt,
                }
            ),
    )

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.id = 0
        data = dict()
        data['id'] = 0
        ok = requests.post(
            f"{self.env_server_base}/create",
            json=data,
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        self.env_id = ok.json()
        self.reward_delta = 0.0
        self.reward_reweight = 0.2
        
    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        data["env_idx"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> Dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> Dict[str, Any]:
        question = self._get("observation")
        return question

    def step(self, payload_str: str, idx) -> StepOutput:
        payload = json.loads(payload_str)
        # 2. 提取 content 和 parameters
        action = payload.get("action_content", "") # 获取实际的内容
        para = payload.get("parameters", {})
        idx = para.get('user_msg_count', 0)
        saved_token_length_count_observation= para.get('saved_token_length_count_observation', 0)
        saved_token_length_count_think = para.get('saved_token_length_count_think', 0)
        token_length_count = para.get('token_length_count', 0)


        # action is the original output of llm
        # print(f"Action: {action}")
        response = self._post("step", {"action": action, "idx": idx})
        # print(response)
         ######## reward shapping ###########
        if response["reward"] > 0:
            if omit_tool_response_num != []:
                self.reward_delta = self.reward_delta + saved_token_length_count_observation/token_length_count
            if "<tool_call><tool_call>" in action_ or "<tool_call>\n<tool_call>" in action_:
                self.reward_delta = self.reward_delta + saved_token_length_count_think/token_length_count
        

        return StepOutput(
            state=response["observation"],
            reward=(1-self.reward_reweight)*response["reward"] +  self.reward_reweight* self.reward_delta,
            done=response["done"],
            skip_turn=response['skip_turn'],
        )

    def reset(self, id: int) -> Dict[str, Any]:
        self.id = id
        response = self._post("reset", {"id": self.id})
        return response
    
    def close(self):
        response = self._post("close", {})
        return response

class SearchQATask(BaseTask):
    env_client_cls = SearchQAEnvClient
    env_name = "SearchQA"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
