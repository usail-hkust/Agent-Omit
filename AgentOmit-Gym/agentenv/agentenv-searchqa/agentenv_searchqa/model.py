from typing import List, Optional

from pydantic import BaseModel

class CreateQuery(BaseModel):
    id: int


class StepQuery(BaseModel):
    env_idx: int
    action: str
    idx: int


class StepResponse(BaseModel):
    observation: str
    reward: float
    done: bool
    skip_turn: list[int]


class ResetQuery(BaseModel):
    env_idx: int
    id: Optional[int] = None


class Task(BaseModel):
    id: str
    content: str
    
class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False
    
class CloseRequestBody(BaseModel):
    env_idx: int