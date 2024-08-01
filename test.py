import joblib
import cv2 as cv
import numpy as np
import zipfile

# Załaduj model z pliku .pkl
with zipfile.ZipFile('random_forest.zip', 'r') as zipf:
    zipf.extractall()

loaded_model = joblib.load('random_forest.pkl')

# Funkcja do wczytania i przetworzenia obrazu
def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (128, 128))  # Zmiana rozmiaru obrazu na 128x128 pikseli
    img = np.array(img) / 255.0       # Normalizacja obrazu
    img = img.reshape(1, -1)          # Przekształcenie obrazu do odpowiedniego kształtu
    return img

# Ścieżka do wskazanego zdjęcia
image_path = 'data/VanGogh/205627.jpg'

# Wczytaj i przetwórz zdjęcie
preprocessed_image = preprocess_image(image_path)

# Użyj modelu do przewidzenia klasy zdjęcia
predicted_class = loaded_model.predict(preprocessed_image)

labels = ['Degas', 'Lorrain', 'Monet', 'Munch', 'Poussin', 'Renoir', 'VanGogh']
# Uzyskaj przewidziany output
print(f"Predicted class: {labels[predicted_class[0]]}")
