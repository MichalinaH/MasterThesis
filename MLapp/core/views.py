from django.shortcuts import render, redirect
from django.http import HttpResponse
import joblib
import cv2 as cv
import numpy as np
import zipfile
import os
from django.conf import settings
from .models import Image
from .forms import ImageForm
from django.urls import reverse
import wikipedia

def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            form.save()
            obj = Image.objects.last()

            image_url = obj.image.url
            image_path = os.path.join(settings.MEDIA_ROOT, obj.image.name)

            with zipfile.ZipFile('../random_forest.zip', 'r') as zipf:
                zipf.extractall()
            loaded_model = joblib.load('../random_forest.pkl')

            def preprocess_image(image_path):
                img = cv.imread(image_path)
                if img is None:
                    raise ValueError(f"Could not read the image from {image_path}")
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img, (128, 128))  # Resize image to 128x128 pixels
                img = np.array(img) / 255.0       # Normalize image
                img = img.reshape(1, -1)          # Reshape image to the correct format
                return img

            try:
                # Process the image
                preprocessed_image = preprocess_image(image_path)

                # Predict the class of the image
                predicted_class = loaded_model.predict(preprocessed_image)

                labels = ['Degas', 'Lorrain', 'Claude Monet', 'Munch', 'Poussin', 'Renoir', 'Van Gogh']
                predicted_artist = f"{labels[predicted_class[0]]}"
                print(predicted_artist)

            except Exception as e:
                return HttpResponse(f"An error occurred: {str(e)}")

            url = reverse('predicted_artist')  # Use the name of your URL pattern
            return redirect(f"{url}?artist={predicted_artist}")


    else:
        form = ImageForm()

    context = {
        "form": form
    }

    return render(request, "core/home.html", context)


def predicted_artist(request):
    artist = request.GET.get("artist")
    print(artist)
    info = wikipedia.summary(artist, sentences = 3, auto_suggest=False)

    image_url = get_infobox_image(artist)

    context = {
        "artist": artist,
        "info": info,
        "image_url": image_url
    }

    return render(request, "core/predicted_artist.html", context)


def get_infobox_image(artist):
    try:
        # Fetch the Wikipedia page for the artist
        page = wikipedia.page(artist, auto_suggest=False)
        
        # Get the list of image URLs
        images = page.images

        # Filter out images that are most likely in the infobox by checking common patterns
        # This is a heuristic; exact selection may require more sophisticated parsing
        infobox_image = None
        for img in images:
            if 'wikimedia' in img or 'thumb' in img:
                infobox_image = img
                break  # Assuming the first match is the infobox image

        return infobox_image

    except wikipedia.exceptions.DisambiguationError as e:
        # Handle disambiguation error
        return None
    except wikipedia.exceptions.PageError:
        # Handle page not found error
        return None
    except Exception as e:
        # Handle any other exceptions
        return None
