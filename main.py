from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


MODEL_PATH = "./model"

app = FastAPI(
    title="Financial Sentiment Analysis API",
    description="API para predecir sentimientos financieros usando un transformer pequeño.",
    version="1.0"
)


class InputData(BaseModel):
    sentence: str


if not os.path.exists(MODEL_PATH):
    raise Exception("No se encontró la carpeta model/. Primero corre: python train_model.py")


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open(os.path.join(MODEL_PATH, "labels.txt"), "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]


@app.get("/")
def home():
    return {
        "message": "Financial Sentiment Analysis API funcionando correctamente"
    }


@app.post("/predict")
def predict(data: InputData):
    inputs = tokenizer(
        data.sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()

    return {
        "sentence": data.sentence,
        "sentiment": labels[predicted_class],
        "confidence": float(probabilities[0][predicted_class])
    }