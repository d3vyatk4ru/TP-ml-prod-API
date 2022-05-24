from pydantic import BaseModel


class PredictResponse(BaseModel):
    id: int
    target: int
