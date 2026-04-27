import os
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


MODEL_NAME = "distilbert-base-uncased"
SAVE_PATH = "./model"


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        padding="max_length",
        truncation=True,
        max_length=128
    )


def train_model():
    print("Descargando dataset de Kaggle...")

    path = kagglehub.dataset_download("sbhatti/financial-sentiment-analysis")

    csv_path = None

    for file in os.listdir(path):
        if file.endswith(".csv"):
            csv_path = os.path.join(path, file)
            break

    if csv_path is None:
        raise FileNotFoundError("No se encontró ningún archivo CSV en el dataset.")

    data = pd.read_csv(csv_path)

    print("Columnas originales encontradas:")
    print(data.columns)

    # Convertimos los nombres de columnas a minúsculas
    data.columns = data.columns.str.strip().str.lower()

    print("Columnas después de limpiar:")
    print(data.columns)

    # Validamos que existan las columnas correctas
    if "sentence" not in data.columns or "sentiment" not in data.columns:
        raise ValueError(
            f"No se encontraron las columnas 'sentence' y 'sentiment'. "
            f"Las columnas disponibles son: {list(data.columns)}"
        )

    data = data[["sentence", "sentiment"]].dropna()

    label_encoder = LabelEncoder()
    data["label"] = label_encoder.fit_transform(data["sentiment"])

    train_df, test_df = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
        stratify=data["label"]
    )

    train_dataset = Dataset.from_pandas(train_df[["sentence", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["sentence", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    test_dataset = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True
    )

    num_labels = len(label_encoder.classes_)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    )

    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    print("Entrenando modelo...")
    trainer.train()

    print("Guardando modelo...")

    os.makedirs(SAVE_PATH, exist_ok=True)

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    labels = list(label_encoder.classes_)

    with open(os.path.join(SAVE_PATH, "labels.txt"), "w", encoding="utf-8") as f:
        for label in labels:
            f.write(label + "\n")

    print("Modelo guardado correctamente en la carpeta model/")


if __name__ == "__main__":
    train_model()