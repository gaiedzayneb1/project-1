import pyttsx3
import os
import time

def text_to_speech(text: str, output_dir: str = "tts_output") -> str:
    if not text.strip():
        raise ValueError("Le texte est vide.")

    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f"audio_{timestamp}.mp3"
    output_path = os.path.join(output_dir, filename)

    engine = pyttsx3.init()
    # Réglages possibles, ex : vitesse, voix
    engine.setProperty('rate', 150)
    # Pour sauvegarder dans un fichier audio
    engine.save_to_file(text, output_path)
    engine.runAndWait()
    engine.stop()

    print(f"[✅ Audio généré localement : {output_path}]")
    return output_path
