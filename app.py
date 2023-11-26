from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
import imageio
import cv2
import numpy as np
import joblib

import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
async def get_root():
    return {"message": "Witaj na stronie głównej!"}

@app.get("/favicon.ico")
async def get_favicon():
    # Tutaj możesz zwrócić plik ikony favicon, jeśli go masz
    # Na przykład: return FileResponse("favicon.ico")
    return {"message": "Brak ikony favicon."}

# Wczytaj model przy użyciu biblioteki joblib
model = joblib.load('model_RESNET50.pkl')

train_input_shape = model.train_input_shape  # Przykładowy atrybut
labels = model.labels

class Obraz(BaseModel):
    obraz: UploadFile

@app.post("/klasyfikuj_obraz")
async def klasyfikuj_obraz(obraz: Obraz):
    # Pobierz zawartość przesłanego obrazu
    image_bytes = await obraz.read()

    # Wczytaj obraz przy użyciu biblioteki imageio
    web_image = imageio.imread(image_bytes)

    # Przy użyciu OpenCV zmień rozmiar obrazu do oczekiwanego rozmiaru
    target_size = (224, 224)  # Przykładowy rozmiar, dostosuj do modelu
    web_image = cv2.resize(web_image, dsize=target_size)

    # Przekształć obraz na format akceptowany przez model
    web_image = web_image.astype(np.float32) / 255.0  # Normalizacja do zakresu [0, 1]

    # Przy użyciu numpy rozszerz wymiar obrazu
    web_image = np.expand_dims(web_image, axis=0)


    # Dokonaj klasyfikacji obrazu
    prediction = model.predict(web_image)
    prediction_probability = np.amax(prediction)
    prediction_idx = np.argmax(prediction)

    predicted_label = labels[prediction_idx].replace('_', ' ')
    prediction_probability_percent = prediction_probability * 100

    print("Predicted artist =", predicted_label)
    print("Prediction probability =", prediction_probability_percent, "%")

    # Wyświetl obraz przy użyciu matplotlib
    plt.imshow(web_image.squeeze())
    plt.axis('off')
    plt.show()

    # Zwróć wynik klasyfikacji w odpowiedzi API
    return {"klasa": predicted_label, "prawdopodobienstwo": prediction_probability_percent}
