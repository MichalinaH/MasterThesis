from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.urls import reverse
from .models import Image
from .forms import ImageForm

import os
import joblib
import cv2 as cv
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image as PilImage
import gzip
import shutil
import wikipedia
import zipfile


LABELS = ['Edgar Degas', 'Claude Lorrain', 'Claude Monet', 'Edvard Munch', 'Nicolas Poussin', 'Auguste Renoir', 'Van Gogh']


def preprocess_image_torch(image_path, device):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = PilImage.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    return img


def preprocess_image_cv(image_path):
    img = cv.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image from {image_path}")
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (128, 128))
    img = np.array(img) / 255.0
    return img.reshape(1, -1)


def decompress_model(gz_path, output_path):
    with gzip.open(gz_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            model_name = form.cleaned_data['model']
            form.save()
            obj = Image.objects.last()
            image_path = os.path.join(settings.MEDIA_ROOT, obj.image.name)

            if model_name == "RANDOM_FOREST":
                try:
                    with zipfile.ZipFile('../random_forest.zip', 'r') as zipf:
                        zipf.extractall()
                    model = joblib.load('../random_forest.pkl')
                    preprocessed_image = preprocess_image_cv(image_path)
                    predicted_class = model.predict(preprocessed_image)[0]
                    predicted_artist = LABELS[predicted_class]
                except Exception as e:
                    return HttpResponse(f"An error occurred: {str(e)}")
                return redirect_to_artist(predicted_artist)

            elif model_name in ["RESNET50", "VGG19"]:
                try:
                    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                    model_path = f"../model_{model_name.lower()}.pth.gz"
                    output_path = f"model_{model_name.lower()}.pth"
                    decompress_model(model_path, output_path)
                    model = load_model(output_path, device)
                    preprocessed_image = preprocess_image_torch(image_path, device)
                    with torch.no_grad():
                        output = model(preprocessed_image)
                        predicted_class = torch.argmax(output, dim=1).item()
                    predicted_artist = LABELS[predicted_class]
                except Exception as e:
                    return HttpResponse(f"An error occurred: {str(e)}")
                return redirect_to_artist(predicted_artist)

    else:
        form = ImageForm()

    return render(request, "core/home.html", {"form": form})


def redirect_to_artist(artist):
    url = reverse('predicted_artist')
    return redirect(f"{url}?artist={artist}")


def predicted_artist(request):
    artist = request.GET.get("artist")
    try:
        info = wikipedia.summary(artist, sentences=3, auto_suggest=False)
    except Exception:
        info = "Information about the artist is not available."

    image_url = get_infobox_image(artist)

    context = {
        "artist": artist,
        "info": info,
        "image_url": image_url,
    }

    return render(request, "core/predicted_artist.html", context)


def get_infobox_image(artist):
    try:
        page = wikipedia.page(artist, auto_suggest=False)
        return page.images[0] if page.images else None
    except (wikipedia.exceptions.DisambiguationError, wikipedia.exceptions.PageError, Exception):
        return None
