# Generated by Django 4.2.7 on 2023-11-01 16:08

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_rename_moddel_image_model'),
    ]

    operations = [
        migrations.RenameField(
            model_name='image',
            old_name='author',
            new_name='artist',
        ),
        migrations.AlterField(
            model_name='image',
            name='model',
            field=models.CharField(choices=[('RANDOM_FOREST', 'Random Forest'), ('RESNET50', 'ResNet50')], max_length=50),
        ),
    ]
