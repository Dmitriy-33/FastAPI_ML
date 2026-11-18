import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import Depends, FastAPI, HTTPException, Request

from app.auth import verify_credentials
from app.schemas import PatientFeatures, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("api")

MODEL_PATH = os.getenv("MODEL_PATH", "diabetes_model.onnx")
THRESHOLD = 0.5

session = None
input_name = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session, input_name
    logger.info("Загружаю модель из %s", MODEL_PATH)
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logger.info("Модель загружена, вход: %s", input_name)
    yield
    logger.info("Останавливаю приложение")


app = FastAPI(title="Diabetes Prediction API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info("Запрос: %s %s", request.method, request.url.path)
    response = await call_next(request)
    logger.info("Ответ: %s %s -> %s", request.method, request.url.path, response.status_code)
    return response


@app.get("/")
def root() -> dict:
    return {"message": "Добро пожаловать в Diabetes Prediction API"}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: PatientFeatures,
    username: str = Depends(verify_credentials),
) -> PredictionResponse:
    data = np.array(
        [[features.Pregnancies, features.Glucose, features.BMI, features.Age]],
        dtype=np.float32,
    )
    try:
        output = session.run(None, {input_name: data})
    except Exception as exc:
        logger.exception("Ошибка инференса")
        raise HTTPException(status_code=500, detail="Ошибка при обработке данных") from exc

    probability = float(output[0][0])
    prediction = int(probability > THRESHOLD)
    logger.info("Пользователь %s: вход=%s, p=%.4f, предсказание=%d", username, data.tolist()[0], probability, prediction)

    return PredictionResponse(prediction=prediction)
