import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 📂 Définition des chemins
chemin_1 = (r"C:\Users\daora\.cache\kagglehub\datasets\andradaolteanu\gtzan-dataset-music-genre-classification"
        r"\versions\1\Data\genres_original")  # Dossier avec sous-dossiers de genres

chemin_2 = r"D:\Transformation_fichier_audio\ressources\fichierwav"  # Dossier contenant uniquement des fichiers WAV

chemin_3 = r"D:\Transformation_fichier_audio\ressources\jazz"
output_path = "mel_spectrograms"  # 📂 Dossier de sortie pour les images

# 📌 Vérifier et créer le dossier de sortie
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 🎨 Fonction pour générer et enregistrer un Mel-Spectrogramme

def save_mel_spectrogram(file_path, save_path):
    try:
        # Charger le fichier audio
        y, sr = librosa.load(file_path, duration=30)
        y, _ = librosa.effects.trim(y)  # Supprimer les silences

        # Générer un Mel-Spectrogramme
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spectrogram_db = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

        # 📌 Générer l'image
        plt.figure(figsize=(2.24, 2.24), dpi=100)
        librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=512, x_axis='time', y_axis='mel', cmap='cool')
        plt.axis('off')  # Enlever les axes

        # 💾 Sauvegarde de l'image
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    except Exception as e:
        print(f"❌ Erreur avec {file_path}: {e}")


# 🔍 1️⃣ Parcourir chemin_1 (avec sous-dossiers de genres)
for genre in os.listdir(chemin_1):
    genre_path = os.path.join(chemin_1, genre)
    output_genre_path = os.path.join(output_path, genre)

    if os.path.isdir(genre_path):  # Vérifier si c'est un dossier
        if not os.path.exists(output_genre_path):
            os.makedirs(output_genre_path)  # 📂 Créer dossier du genre

        for file in tqdm(os.listdir(genre_path), desc=f"🔍 Génération {genre}"):
            file_path = os.path.join(genre_path, file)
            name_file = file.split(".w")[0]
            if file.endswith(".wav"):  # Vérifier format audio
                save_path = os.path.join(output_genre_path, f"{name_file}.png")  # Nom du fichier image
                save_mel_spectrogram(file_path, save_path)

# 🔍 2️⃣ Parcourir chemin_2 (uniquement des fichiers WAV)
output_wav_path = os.path.join(output_path, "zouk")  # 📂 Dossier spécifique pour ces fichiers
if not os.path.exists(output_wav_path):
    os.makedirs(output_wav_path)

for file in tqdm(os.listdir(chemin_2), desc="🔍 Génération fichiers WAV seuls"):
    file_path = os.path.join(chemin_2, file)
    if file.endswith(".wav"):
        name_file = file.split(".w")[0]
        save_path = os.path.join(output_wav_path, f"{name_file}.png")
        save_mel_spectrogram(file_path, save_path)

# 🔍 3 Parcourir chemin_3 (uniquement des fichiers WAV)

for file in tqdm(os.listdir(chemin_3), desc="🔍 Génération fichiers WAV seuls"):
    file_path = os.path.join(chemin_3, file)
    if file.endswith(".wav"):
        name_file = file.split(".w")[0]
        save_path = os.path.join(output_wav_path, f"{name_file}.png")
        save_mel_spectrogram(file_path, save_path)

print("✅ Génération terminée ! Toutes les images sont dans 'mel_spectrograms/'")


"""import kagglehub
import os

path = kagglehub.dataset_download("daoudarafioustphane/updatamusic")

print(list(os.listdir(f'{path}/mel_spectrograms/')))

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Définir le chemin principal
path = "/kaggle/input/updatamusic"  # Remplacez par votre chemin
# Lister les sous-dossiers du chemin principal
subfolders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
print(subfolders)
# Pour chaque sous-dossier, afficher les images
for subfolder in subfolders:
    # Construire le chemin complet vers le sous-dossier
    subfolder_path = os.path.join(path, subfolder)

    # Lister les fichiers dans le sous-dossier
    files = os.listdir(subfolder_path)
    print(files)

    print(f"Fichiers dans le dossier {files}:")

    # Afficher les images dans ce sous-dossier
    for file_name in files:
        image_path = os.path.join(subfolder_path, file_name)
        print(image_path)
        subfolder_path2 = os.listdir(image_path)"""
"""       for imag in subfolder_path2:
            image_path2 = os.path.join(image_path, imag)
            print(imag)
            print(image_path2)
            img = mpimg.imread(image_path2)
            # Afficher l'image
            plt.figure(figsize=(5, 5))
            plt.imshow(img)
            plt.title(f"{subfolder} - {file_name}")  # Afficher le nom du dossier et de l'image
            plt.axis('off')  # Masquer les axes pour mieux visualiser l'image
            plt.show()"""
"""dataset_path = f'{path}/mel_spectrograms/'"""