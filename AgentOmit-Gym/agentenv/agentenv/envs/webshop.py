import json
from typing import Any, Mapping
import re
import requests
from typing import List, Dict, Optional
import json

from agentenv.controller import (
    BaseAdapter,
    BaseEnvClient,
    BaseTask,
    extract_python_code_blocks,
    format_code_as_action_prompt,
    format_function_call_prompt,
    parse_python_code_comments,
)
from agentenv.controller.types import (
    ActionFormat,
    ActionWithTought,
    ConversationMessage,
    StepOutput,
)

webshopping_sys_prompt = """You are web shopping. I will give you instructions about what to do. You have to follow the instructions. Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction. You can use search action if search is available. You can click one of the buttons in clickables.

Your output must strictly follow this format:
<think>your thoughts.</think>
<tool_call>your next action.</tool_call>
<tool_response_N>your next action.</tool_response_N>
<think>your thoughts.</think>
...(continue to generate <tool_call>...</tool_call> for problem-solving, or generate <answer>...</answer> if the task is finished)...
<answer>...</answer>

Reminder: 
1. An action should be wrapped in <tool_call>...</tool_call>, and the action content should be the following structure: search[keywords] or click[value]
2. If the action is not valid, perform nothing. Keywords in search are up to you, but the value in click must be a value in the list of available actions.
3. Remember that your keywords in search should be carefully designed.
"""

efficient_webshopping_sys_prompt = """You are web shopping. I will give you instructions about what to do. You have to follow the instructions. Every round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction. You can use search action if search is available. You can click one of the buttons in clickables.

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
1. An action should be wrapped in "<tool_call>...</tool_call>", and the action content should be the following structure: search[keywords] or click[value]
2. If the action is not valid, perform nothing. Keywords in search are up to you, but the value in click must be a value in the list of available actions.
3. Remember that your keywords in search should be carefully designed.
4. "<think></think>" is a good way to save context when you are confident about your next action.
5. "<omit_tool_response_N></omit_tool_response_N>" can help you save context by omitting prior tool responses at turn N, you are encouraged to use when there have too many turns or are clearly stuck on a given step.

Let's start! Do remember to derive "<think> </think>" or "<omit_tool_response_N></omit_tool_response_N>" when necessary to save context!

"""


WEBSHOP_FUNCTION_DESCRIPTION = [
    {
        "name": "search",
        "description": "If the search bar is on the page, you can use this function to search for a product. If the action is not valid, perform nothing.",
        "parameters": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "string",
                    "description": "Keywords in search are up to you. Remember that your keywords in search should be carefully designed.",
                }
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "click",
        "description": "Click on a button.",
        "parameters": {
            "type": "object",
            "properties": {
                "item": {
                    "type": "string",
                    "description": "The item to click. The item should be one of the cilickable values on the page.",
                }
            },
            "required": ["item"],
        },
    },
]

class WebshopAdapter(BaseAdapter):
    conversation_start_dict = {
        ActionFormat.REACT: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": efficient_webshopping_sys_prompt,
                }
            ),
        ),
        ActionFormat.FUNCTION_CALLING: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nAn action should be done by invoking a function.\n\n{format_function_call_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}\n\n\nIf the page remains unchanged, it might indicate that your action is invalid.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
        ActionFormat.CODE_AS_ACTION: (
            ConversationMessage(
                {
                    "from": "human",
                    "loss": None,
                    "value": f"You are web shopping.\nI will give you instructions about what to do.\nYou have to follow the instructions.\nEvery round I will give you an observation and a list of available actions, you have to respond an action based on the state and instruction.\nYou can use search action if search is available.\nYou can click one of the buttons in clickables.\nYou can perform one of these actions by writing python code to invoke a function.\n\n{format_code_as_action_prompt(WEBSHOP_FUNCTION_DESCRIPTION)}\n\n\nIf the page remains unchanged, it might indicate that your action is invalid.",
                }
            ),
            ConversationMessage({"from": "gpt", "loss": False, "value": "Ok."}),
        ),
    }
    @staticmethod
    def parse_react(text):
        invalid_format_flg = False
        _split = text.rsplit("<tool_call>", 1)
        if len(_split) == 2:
            if "search[" in text or "click[" in text:
                _thought, _action = _split[0], _split[1].replace("</tool_call>", "")
            else:
                _thought, _action = _split[0], ""
        else:
            invalid_format_flg = True
            _thought = text
            _action = ""

        thought = _thought.strip()
        action = _action.strip()
        if invalid_format_flg:
            print(
                "The text is not in the correct format. Parsing result may not be accurate."
            )
            print("###RAW TEXT:\n", text)
            print("\n###PARSED THOUGHT:\n", thought)
            print("\n###PARSED ACTION:\n", action)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def parse_function_calling(text: str) -> ActionWithTought:
        _fn_call = json.loads(
            "{" + text.split("{", 1)[-1].rsplit("}", 1)[0] + "}", strict=False
        )
        thought = _fn_call["thought"]
        fn_name = _fn_call["function_name"]
        args = _fn_call["arguments"]
        if fn_name not in ["search", "click"]:
            raise ValueError("Invalid function name.")
        if fn_name == "search":
            action = f"search[{args['keywords']}]"
        else:
            action = f"click[{args['item']}]"
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_function_calling(action_with_thought: ActionWithTought) -> str:
        if action_with_thought.action.startswith("search"):
            fn_name = "search"
            args = {"keywords": action_with_thought.action.split("[")[-1].split("]")[0]}
        elif action_with_thought.action.startswith("click"):
            fn_name = "click"
            args = {"item": action_with_thought.action.split("[")[-1].split("]")[0]}
        else:
            raise ValueError("Invalid action.")
        return json.dumps(
            {
                "thought": action_with_thought.thought,
                "function_name": fn_name,
                "arguments": args,
            },
            ensure_ascii=False,
            indent=2,
        )

    @staticmethod
    def parse_code_as_action(text: str) -> ActionWithTought:
        def search(keywords: str):
            return f"search[{keywords}]"

        def click(item: str):
            return f"click[{item}]"

        text = extract_python_code_blocks(text)
        try:
            action = eval(text, {}, {"search": search, "click": click})
        except Exception as e:
            raise ValueError("Invalid action.") from e
        thought = parse_python_code_comments(text)
        return ActionWithTought(thought=thought, action=action)

    @staticmethod
    def to_code_as_action(action_with_thought: ActionWithTought) -> str:
        text = f"```python\n# {action_with_thought.thought}\n"
        if action_with_thought.action.startswith("search"):
            text += f"search({repr(action_with_thought.action.split('[')[-1].split(']')[0])})"
        elif action_with_thought.action.startswith("click"):
            text += f"click({repr(action_with_thought.action.split('[')[-1].split(']')[0])})"
        text += "\n```"
        return text

    @staticmethod
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


class WebshopEnvClient(BaseEnvClient):
    adapter_cls = WebshopAdapter

    def __init__(
        self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(
            f"{self.env_server_base}/create",
            timeout=self.timeout,
        )
        if ok.status_code != 200:
            raise requests.RequestException(f"Failed to create environment: {ok}")
        self.conversation_start = self.adapter_cls.conversation_start_dict[
            self.action_format
        ]
        self.env_id = ok.json()
        self.turn = 0
        self.reward_delta = 0.0
        self.reward_reweight = 0.2

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["env_idx"] = self.env_id
        max_retries = 5
        for attempt in range(max_retries):
            res = requests.post(
                f"{self.env_server_base}/{path}",
                json=data,
                timeout=self.timeout,
            )
            if res.status_code == 503:
                import time

                time.sleep(0.1)
            elif res.status_code == 200:
                break
            else:
                print("---------------------")
                print(res.status_code)
                print(data)
        assert res.status_code == 200
        return res.json()

    def _get(self, path: str) -> dict[str, Any]:
        res = requests.get(
            f"{self.env_server_base}/{path}?env_idx={self.env_id}",
            timeout=self.timeout,
        )
        assert res.status_code == 200
        return res.json()

    def observe(self) -> dict[str, Any]:
        response = self._get("observation")
        return response

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
        omit_tool_response_num = WebshopAdapter.parse_tool_response_skip(action_)

        if action_.endswith("</s>"):
            action = action_[:-5]
        else:
            action = action_
        try:
            action = WebshopAdapter.action_parser(action, self.action_format)
        except Exception as e:
            print(e, action)
            return StepOutput(
                state=f"<tool_response_{self.turn}>Invalid Action.</tool_response_{self.turn}>", reward=0.0, done=False
            )
        # print('===debug 0===')
        # print(action)
        # print('===debug 0===')
        response = self._post("step", {"action": action})
        # print('===debug 1===')
        # print(response)
        # print('===debug 1===')
        # last_sep_idx = response['state'].rindex('[SEP]')
        # response_without_instruction =  response[last_sep_idx:]
        first_sep = response['state'].index('[SEP]')
        second_sep = response['state'].index('[SEP]', first_sep + 1)
        response_without_instruction = response['state'][second_sep:]
        formatted_observation = f"<tool_response_{self.turn}>{response_without_instruction}</tool_response_{self.turn}>"
        # print('===debug 2===')
        # print(formatted_observation)
        # print('===debug 2===')


        ######## reward shapping ###########
        if response["reward"] > 0:
            if omit_tool_response_num != []:
                self.reward_delta = self.reward_delta + saved_token_length_count_observation/token_length_count
            if "<tool_call><tool_call>" in action_ or "<tool_call>\n<tool_call>" in action_:
                self.reward_delta = self.reward_delta + saved_token_length_count_think/token_length_count
        
        if '<answer>' in action_:
            return StepOutput(
                state=formatted_observation,
                reward=(1-self.reward_reweight)*response["reward"] +  self.reward_reweight* self.reward_delta,
                done=True,
                skip_turn=omit_tool_response_num,
            )
        else:
            return StepOutput(
                state=formatted_observation,
                reward=response["reward"],
                done=response["done"],
                skip_turn=omit_tool_response_num,
            )

    def reset(self, idx: int) -> dict[str, Any]:
        response = self._post("reset", {"session_id": idx})
        response[0] = self.observe()
        return response

    def close(self):
        response = self._post("close", {})



class WebshopTask(BaseTask):
    env_client_cls = WebshopEnvClient
    env_name = "WebShop"

    def __init__(
        self,
        client_args: Mapping[str, Any] | Mapping[str, Any],
        n_clients: int,
        *args,
        **kwargs,
    ):
        super().__init__(client_args, n_clients, *args, **kwargs)
