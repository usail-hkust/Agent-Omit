from typing import Any, Mapping
import json
import re
from typing import List, Dict, Optional
import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


text_carft_sys_prompt = """You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format \"craft [target object] using [input ingredients]\".
Every round I will give you an observation, you have to respond an action based on the state and instruction. You can \"get\" an object (ingredients) from the inventory or the environment, look-up the game inventory by \"inventory\", or \"craft\" (target) using any of the crafting commands.

Your output must strictly follow this format:
<think>your thoughts.</think>
<tool_call>your next action.</tool_call>
<tool_response>your next action.</tool_response>
<think>your thoughts.</think>
...(continue to generate <tool_call>...</tool_call> for problem-solving, or generate <answer>...</answer> if the task is finished)...
<answer>...</answer>

Reminder: 
1. Always specify the quantity when using \"get\" and \"craft\" commands. - Example of get: <tool_call>get 1 lapis lazuli</tool_call> - Example1 of craft: <tool_call>craft 1 blue dye using 1 lapis lazuli</tool_call> - Example2 of craft: <tool_call>craft 1 golden carrot using 8 gold nugget, 1 carrot</tool_call>
2. When using \"get\" command, do not specify whether the item comes from the inventory or the environment.
3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like \"planks\", you can use special types of the same ingredient e.g. \"dark oak planks\" in the command instead..

"""

efficient_text_carft_sys_prompt = """You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format \"craft [target object] using [input ingredients]\".
Every round I will give you an observation, you have to respond an action based on the state and instruction. You can \"get\" an object (ingredients) from the inventory or the environment, look-up the game inventory by \"inventory\", or \"craft\" (target) using any of the crafting commands.

You use "<think>your thoughts.</think>" for in-depth thinking process; or user "<think></think>" if you think you can directly generate correct tool call action without any thinking process. 
You use "<tool_call>your next action.</tool_call>" for next action; or use "<omit_tool_response_N></omit_tool_response_N><tool_call>your next action.</tool_call>" to simultaneously generate the next action and omit prior tool responses at turn N to save context.

Your output must strictly follow this format:
<think>your thoughts.</think> (or <think></think>)
<tool_call>your next action.</tool_call> (or <omit_tool_response_N></omit_tool_response_N><tool_call>your next action.</tool_call>)
<tool_response_N>your observation after invoking tool.</tool_response_N> (or <omitted_tool_response_N></omitted_tool_response_N>)
<think>your thoughts.</think> (or <think></think>)
...(continue to generate tool_call for problem-solving, or generate <answer>...</answer> if the task is finished)...
<answer>...</answer>

Reminder: 
1. Always specify the quantity when using \"get\" and \"craft\" commands. - Example of get: <tool_call>get 1 lapis lazuli</tool_call> - Example1 of craft: <tool_call>craft 1 blue dye using 1 lapis lazuli</tool_call> - Example2 of craft: <tool_call>craft 1 golden carrot using 8 gold nugget, 1 carrot</tool_call>
2. When using \"get\" command, do not specify whether the item comes from the inventory or the environment.
3. You can use ONLY crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like \"planks\", you can use special types of the same ingredient e.g. \"dark oak planks\" in the command instead..
4. "<think></think>" is a good way to save context when you are confident about your next action.
5. "<omit_tool_response_N></omit_tool_response_N>" can help you save context by omitting prior tool responses at turn N, you are encouraged to use when there have too many turns or are clearly stuck on a given step.

Let's start! Do remember to derive "<think> </think>" or "<omit_tool_response_N></omit_tool_response_N>" when necessary to save context!

"""


class TextCraftEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage(
            {
                "from": "human",
                "loss": None,
                "value": efficient_text_carft_sys_prompt,
            }
        ),
    )

    def __init__(
        self,
        env_server_base: str,
        data_len: int,
        *args,
        timeout: int = 300,
        minecraft_dir: str = "agentenv_textcraft/",
        commands: str = None,
        goal: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len
        self.turn = 0
        self.reward_delta = 0
        self.reward_reweight = 0.2
        dir_info = {"minecraft_dir": minecraft_dir, "commands": commands, "goal": goal}
        ok = requests.post(
            f"{self.env_server_base}/create", timeout=self.timeout, json=dir_info
        )
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")

        ok = ok.json()
        self.env_id = ok["id"]
        self.info = {
            "observation": ok["observation"],
            "reward": 0,
            "done": False,
        }

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(
            f"{self.env_server_base}/{path}",
            json=data,
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?id={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def parse_tool_response_skip(self, text: str) -> List[int]:
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


    def step(self, payload_str: str, idx) -> StepOutput:
        payload = json.loads(payload_str)
        # 2. 提取 content 和 parameters
        action_ = payload.get("action_content", "") # 获取实际的内容
        para = payload.get("parameters", {})
        idx = para.get('user_msg_count', 0)
        saved_token_length_count_observation= para.get('saved_token_length_count_observation', 0)
        saved_token_length_count_think = para.get('saved_token_length_count_think', 0)
        token_length_count = para.get('token_length_count', 0)


        self.turn = int(idx)
        omit_tool_response_num = self.parse_tool_response_skip(action_)

        #action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        action_matches = re.findall(r"<tool_call>(.*?)</tool_call>", action_, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state="Error: Only one 'Action' is allowed per response. Please adjust your response.",
                reward=0,
                done=False,
            )
        action = action_matches[-1] if action_matches else ""
        action = re.sub(r"[^A-Za-z0-9, ]+", "", action)
        action = " ".join(action.split()).strip()
        response = self._post("step", {"action": action})
        
        formatted_observation = f"<tool_response_{self.turn}>{response['observation']}</tool_response_{self.turn}>"
        
        self.info = {
            "observation": formatted_observation,
            "reward": response["reward"],
            "done": response["done"],
            "skip_turn": omit_tool_response_num,
        }

        if response["reward"] > 0:
            if omit_tool_response_num != []:
                self.reward_delta = self.reward_delta + saved_token_length_count_observation/token_length_count
            if "<tool_call><tool_call>" in action_ or "<tool_call>\n<tool_call>" in action_:
                self.reward_delta = self.reward_delta + saved_token_length_count_think/token_length_count
        

        return StepOutput(
            state=formatted_observation,
            reward=(1-self.reward_reweight)*response["reward"] +  self.reward_reweight* self.reward_delta,
            done=response["done"],
            skip_turn=omit_tool_response_num,
        )

    def reset(self, idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": idx})
        self.info.update(
            {
                "observation": response["observation"],
                "reward": 0,
                "done": False,
                "skip_turn": []
            }
        )
        return response
   
    def close(self):
        response = self._post("close",{})
        return response

class TextCraftTask(BaseTask):
    env_client_cls = TextCraftEnvClient
    env_name = "TextCraft"

    def __init__(
        self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs
    ) -> None:
        super().__init__(client_args, n_clients, *args, **kwargs)
