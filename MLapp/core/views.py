from django.shortcuts import render, redirect
from django.http import HttpResponse

from .forms import ImageForm

def home(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.cleaned_data['image']
            print(image)  
            return redirect("home")

    else:
        form = ImageForm()

    context = {
        "form": form
    }
    return render(request, "core/home.html", context)