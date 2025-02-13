import os
import ffmpeg

# Dossiers d'entrée et de sortie
input_dir = "ressources/fichiermp4"
output_dir = "ressources/fichierwav"

# S'assurer que le dossier de sortie existe
os.makedirs(output_dir, exist_ok=True)


# Fonction pour générer le prochain nom de fichier avec l'incrémentation
def get_next_filename(output_dir):
    existing_files = [f for f in os.listdir(output_dir) if f.startswith("zouk.") and f.endswith(".wav")]

    if not existing_files:
        return "zouk.00000.wav"

    # Extraire les numéros existants et trouver le max
    numbers = [int(f[5:10]) for f in existing_files if f[5:10].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1

    return f"zouk.{next_number:05d}.wav"


# Lister tous les fichiers dans le dossier d'entrée
for filename in os.listdir(input_dir):
    if filename.endswith(".mp4"):  # Vérifier si le fichier est un fichier .mp4
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, get_next_filename(output_dir))

        try:
            # Convertir les 30 premières secondes du fichier vidéo en fichier audio WAV
            ffmpeg.input(input_file, t=30).output(output_file, acodec='pcm_s16le', ac=1, ar='44100').run(
                overwrite_output=True)
            print(f"Conversion terminée : {input_file} -> {output_file}")
        except Exception as e:
            print(f"Erreur lors de la conversion de {input_file} : {e}")
