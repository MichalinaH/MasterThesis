from django.db import models



class Image(models.Model):

    MODELS = [
    ("RANDOM_FOREST", "Random Forest"),
    ("RESNET50", "ResNet50"),
    ("VGG19", "Vgg19")
    ]

    artist = models.CharField(max_length=50, blank=True, null=True)
    image = models.ImageField(upload_to="images")
    model = models.CharField(max_length=50, choices=MODELS)

    def __str__(self) -> str:
        return self.artist