# Generated by Django 4.2.7 on 2023-11-01 15:51

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_image_author_image_moddel'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='moddel',
            new_name='model',
        ),
    ]
