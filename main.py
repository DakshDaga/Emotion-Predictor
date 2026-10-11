import re
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from keras.models import load_model
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware




model_path = "Artifacts/BiGRU_model.keras"
vectorizer_path = "Artifacts/vectorizer.pkl"

emotion_labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

emotion_emojis = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}

# Request and Response schemas

class Text_input(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel happy and excited"}
    )

class Predicted_response(BaseModel):
    input_text: str
    predicted_emotion: str
    confidence: float
    all_probs: dict[str, float]


class Health_response(BaseModel):
    status: int
    model_loaded: bool



# Model loading and lifespan management

dl_model = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Loading the model and tokenizer...')

    dl_model["model"] = load_model(model_path)
    with open(vectorizer_path, 'rb') as f:
        dl_model['vectorizer'] = pickle.load(f)
    print("Model loaded successfully")

    yield

    dl_model.clear()



# Mount the static files to FastAPI
# Enable CORS (Cross-Origin Resource Sharing) to allow requests from different origins.

app = FastAPI(
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")



# API Endpoints
# A. Server UI at homepage ('/')
# B. Health Check Endpoint ('/health')
# C. Predict Emotion Endpoint ('/predict')

@app.get("/", include_in_schema=False)
def get_ui():
    return FileResponse("static/index.html")


@app.get("/health", response_model= Health_response)
def get_health():
    return Health_response(status="Server is working", model_loaded=bool(dl_model))


@app.post("/predicct", response_model = Predicted_response)
def predict_emotion(text_input: Text_input):
    model = dl_model.get("model")
    vectorizer = dl_model.get("vectorizer")

    if model == None or vectorizer == None:
        raise HTTPException(status_code=503, detail="Model isn't loaded yet.. Pls try again later")

    text_sequence = vectorizer([text_input.text]).numpy()
    probabilities = model.predict(text_sequence)[0]

    top_emotion_idx = np.argmax(probabilities)
    all_probs = {
        label: prob for label, prob in zip(emotion_labels, probabilities)
    }

    return Predicted_response(
        input_text= text_input.text,
        predicted_emotion= emotion_labels[top_emotion_idx],
        confidence= float(probabilities[top_emotion_idx]),
        all_probs= all_probs
    )
