from pydantic import BaseModel

class CalcRequest(BaseModel):
    first: float
    second: float
    operation: str

class CalcResponse(BaseModel):
    first: float
    second: float
    operation: str
    response: float