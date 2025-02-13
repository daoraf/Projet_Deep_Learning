import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# 📂 Définition des chemins
dataset_path = "mel_spectrograms/"
img_size = (224, 224)  # Taille des images pour VGG16
batch_size = 32
num_classes = len(os.listdir(dataset_path))  # Nombre de genres musicaux (ou classes)

# 📌 Générateur d'images pour augmenter les données
datagen = ImageDataGenerator(
    rescale=1.0/255,  # Normalisation
    # rotation_range=20,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# 🔄 Chargement des données
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 🔥 Chargement de VGG16 sans la dernière couche
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # On freeze les couches pré-entraînées

# 🔧 Ajout de notre propre classificateur
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation='softmax')(x)  # Classification multi-classes

# 🎯 Création du modèle final
model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# 📊 Entraînement du modèle
epochs = 10
history = model.fit(train_generator, validation_data=val_generator, epochs=epochs)

# 📌 Sauvegarde du modèle
model.save("music_genre_classification_vgg16.h5")

# 📈 Affichage des performances
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss")

plt.show()
