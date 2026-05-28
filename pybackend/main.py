from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import CalcRequest, CalcResponse
from calc import calc

app = FastAPI()
app.add_middleware(
     CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/calc", response_model=CalcResponse)
def calculateEndpoint(request: CalcRequest):
    try:
        result = calc(request.first, request.second, request.operation)
        return CalcResponse(
            response = result,
            first = request.first, 
            second = request.second,
            operation = request.operation
        )
    except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error))