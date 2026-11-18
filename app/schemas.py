from pydantic import BaseModel, Field, StrictFloat, StrictInt


class PatientFeatures(BaseModel):

    Pregnancies: StrictInt = Field(ge=0, le=20, description="Количество беременностей")
    Glucose: StrictFloat = Field(gt=0, le=300, description="Уровень глюкозы")
    BMI: StrictFloat = Field(gt=0, le=100, description="Индекс массы тела")
    Age: StrictInt = Field(ge=1, le=120, description="Возраст")


class PredictionResponse(BaseModel):

    prediction: int
