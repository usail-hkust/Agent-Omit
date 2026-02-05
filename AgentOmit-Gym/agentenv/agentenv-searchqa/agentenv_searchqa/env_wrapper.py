"""
SearchQAEnvServer
"""


from typing import Optional
import threading
import json
import os
import re
import argparse
import datasets
from typing import List, Dict, Optional

from .utils import Config
from .retriever import get_retriever
from .reward_score import compute_score_em, compute_score_em_format

file_path = os.path.dirname(os.path.abspath(__file__))

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NotInitializedError(Exception):
    pass

TEST_ITEM_RANGE = {
    "nq": (0, 3610),
    "triviaqa": (3610, 14923),
    "popqa": (14923, 29190),
    "hotpotqa": (29190, 36595),
    "2wikimultihopqa": (36595, 49171),
    "musique": (49171, 51588),
    "bamboogle": (51588, 51713),
}

TRAIN_ITEM_RANGE = {
    "nq": (51713, 130881),
    "hotpotqa": (130881, 221328),
}

ITEM_RANGE = {
    "test": (0, 51713),
    "train": (51713, 221328),
}

faiss_gpu = os.environ.get("SEARCHQA_FAISS_GPU", "False").lower() == "true"
retrieval_method = os.environ.get("SEARCHQA_RETRIEVAL_METHOD", "e5")
retrieval_topk = int(os.environ.get("SEARCHQA_RETRIEVAL_TOPK", "3"))
index_path = os.environ.get(
    "SEARCHQA_INDEX_PATH",
    os.path.join(file_path, "..", "retrieve_data", "e5_Flat.index"),
)
corpus_path = os.environ.get(
    "SEARCHQA_CORPUS_PATH",
    os.path.join(file_path, "..", "retrieve_data", "wiki-18.jsonl"),
)
retrieval_model_path = os.environ.get(
    "SEARCHQA_RETRIEVAL_MODEL_PATH",
    os.path.join(file_path, "..", "retrieve_data", "e5-base-v2"),
)
retrieval_use_fp16 = (
    os.environ.get("SEARCHQA_RETRIEVAL_USE_FP16", "True").lower() == "true"
)
retrieval_batch_size = int(os.environ.get("SEARCHQA_RETRIEVAL_BATCH_SIZE", "512"))


class SearchQAEnvServer:
    """
    SearchQAEnvServerEnvServer
    """

    def __init__(self) -> None:

        config = Config(
            retrieval_method=retrieval_method,  # or "dense"
            index_path=index_path,
            corpus_path=corpus_path,
            retrieval_topk=retrieval_topk,
            faiss_gpu=faiss_gpu,
            retrieval_model_path=retrieval_model_path,
            retrieval_pooling_method="mean",
            retrieval_query_max_length=256,
            retrieval_use_fp16=retrieval_use_fp16,
            retrieval_batch_size=retrieval_batch_size,
        )

        self.retriever = get_retriever(config)
        self._max_id = 0
        self.env = {}
        self.ls = []
        self._lock = threading.Lock()
        self.turn = 0
        train_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(file_path, "queries", "train.parquet"),
            keep_in_memory=False,
        )["train"]
        test_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.join(file_path, "queries", "test.parquet"),
            keep_in_memory=False,
        )["train"]

        self.dataset = {
            "test": test_dataset,
            "train": train_dataset,
        }

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

    def create(self, item_id: int = 0) -> int:
        with self._lock:
            env_idx = self._max_id
            self._max_id += 1

        self.env[env_idx] = self._fetch_data(
            item_id
        )  # redundancy fetch to prevent NoneType Error
        self.ls.append(env_idx)

        return env_idx

    def step(self, env_idx, response: str, idx):
        """
        Perform a step in the environment with the given action.
        Input:
            env_idx: the index of the environment
            action: a string in the format "<search> query </search>" or "<answer> answer </answer>"
        Output:
            observation: the observation after taking the action
            reward: the reward received after taking the action
            done: whether the episode is done
            info: additional information (not used here)
        Raises:
            ValueError: if the action is not a valid string format
        """
        self.turn = int(idx)
        omit_tool_response_num = self.parse_tool_response_skip(response)

        self._check_env_idx(env_idx)
        reward = 0
        done = False
        observation = ""

        if isinstance(response, str):  # for llm output
            pattern = r"<(tool_call|answer)>(.*?)</\1>"
            match = re.search(pattern, response, re.DOTALL)
            if match:
                content = match.group(2).strip()
                action = match.group(1)
            else:
                content = ""
                action = None
        else:
            raise ValueError(f"Invalid action type: {type(action)}")

        if action == "tool_call":
            search_query = content
            logger.info(f"Search query: {search_query}")
            search_results = self._search(search_query)
            # search_results = ["pass"]
            observation = f"<tool_response_{self.turn}>{search_results.pop(0).strip()}</tool_response_{self.turn}>"
        elif action == "answer":
            # Check if the answer is correct
            # format_score = compute_score_em_format(
            #     response,
            #     self.env[env_idx]["reward_model"]["ground_truth"],
            # )
            # print(f"Format score: {score_format}")
            score = compute_score_em(
                solution_str=response,
                ground_truth=self.env[env_idx]["reward_model"]["ground_truth"],
            )
            # print(f"SubEM score: {score}")
            reward = score
            if score == 1:
                done = True
                observation = (
                    f"<tool_response_{self.turn}>Congratulations! You have answered the question correctly.</tool_response_{self.turn}>"
                )
            else:
                done = True
                observation = f"<tool_response_{self.turn}>Sorry, your answer is incorrect.</tool_response_{self.turn}>"
        else:
            observation = f"<tool_response_{self.turn}>Your previous action is invalid. If you want to search, you should put the query between <search> and </search>. If you want to give the final answer, you should put the answer between <answer> and </answer>. Please try again.</tool_response_{self.turn}>"
        return observation, reward, done, omit_tool_response_num

    def observation(self, env_idx):
        self._check_env_idx(env_idx)
        question = self.env[env_idx]["question"]
        user_prompt = f"""Question: {question.strip()}"""
        return user_prompt

    def reset(self, env_idx, item_id: Optional[int] = None):
        self._check_env_idx(env_idx)
        self.env[env_idx] = self._fetch_data(item_id)

    def _search(self, search_query: str):
        results, scores = self.retriever.search(
            query=[search_query], num=3, return_score=True
        )
        # Format response
        resp = []
        combined = []
        for doc, score in zip(results, scores):
            combined.append({"document": doc, "score": score})
        resp.append(combined)
        result = [self._passages2string(r) for r in resp]
        logger.info(f"Search results: {result}\nRAW: {resp}")
        return result

    def _passages2string(self, retrieval_result):
        format_reference = ""
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item["document"]["contents"]
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference

    def _check_env_idx(self, env_idx):
        if env_idx not in self.env:
            raise IndexError(f"Env {env_idx} not found")
        if self.env[env_idx] is None:
            raise NotInitializedError(f"Env {env_idx} not initialized")

    def _fetch_data(self, item_id: int):
        """
        Fetch data from the dataset based on the item_id.
        """
        _id = None
        for mode, r in ITEM_RANGE.items():
            if r[0] <= item_id < r[1]:
                _id = item_id - r[0]

                return self.dataset[mode][_id]
        if _id is None:
            raise ValueError(f"Item id {item_id} is out of range.")
        
    def __del__(self):
        for idx in self.ls:
            del self.env[idx]
            print(f"-------Env {idx} closed--------")
    def close(self,id):
        try:
            self._check_env_idx(id)
            self.ls.remove(id)
            del self.env[id]
            print(f"-------Env {id} closed--------")
            return True
        except Exception as e:
            print(f"Error closing env {id}: {e}")
            return False

searchqa_env_server = SearchQAEnvServer()
