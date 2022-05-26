from typing import (
    List,
    Union,
)

from pydantic import ( 
    BaseModel,
    conlist,
)


class PredictResponse(BaseModel):
    id: int
    target: int


class InputDataRequest(BaseModel):
    data: List[conlist(Union[float, str, None])]
    features: List[str]
