"""
FastAPI app: POST /predict takes raw customer details and returns
churn probability + Churn / No Churn using the saved pipeline and threshold.

Run from the Telco folder:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_artifact, predict_one

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: Optional[float] = Field(
        default=None, ge=0,
        description="Optional. If omitted, estimated as tenure x MonthlyCharges.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "No",
                "Dependents": "No", "tenure": 2, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "No", "StreamingMovies": "No",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.7, "TotalCharges": 151.65,
            }
        }
    }


class Prediction(BaseModel):
    churn_probability: float
    prediction: Literal["Churn", "No Churn"]
    threshold: float
    total_charges_estimated: bool


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifact()          # load the model once at startup, not per request
    yield


app = FastAPI(title="Telco Churn API", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    art = load_artifact()
    return {"status": "ok", "threshold": art["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    data = customer.model_dump()
    estimated = data["TotalCharges"] is None
    if estimated:
        data["TotalCharges"] = round(data["tenure"] * data["MonthlyCharges"], 2)
    try:
        result = predict_one(data)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {**result, "total_charges_estimated": estimated}