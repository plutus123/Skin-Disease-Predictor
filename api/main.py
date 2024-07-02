from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("/Users/asthamishra/Desktop/Projects/Skin Disease Prediction/saved_model/my_model.h5")

CLASS_NAMES = [
    "Eczema",
    "Atopic Dermatitis",
    "Basal Cell Carcinoma",
    "Melanocytic Nevi",
    "Melanoma",
    "Psoriasis",
    "Seborrheic Dermatitis",
    "Tinea (Ringworm)",
    "Candidiasis",
    "Warts",
    "Molluscum Contagiosum"
]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    # Open the image using PIL
    image = Image.open(BytesIO(data))
    # Convert the image to RGB if it is not already (important for grayscale images)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Resize the image while maintaining the aspect ratio
    image = image.resize((224, 224))
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image_data = await file.read()
    image = read_file_as_image(image_data)
    img_batch = np.expand_dims(image, 0)  # Shape (1, 224, 224, 3)

    # Make prediction
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
