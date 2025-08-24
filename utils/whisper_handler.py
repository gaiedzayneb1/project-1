import whisper
import os
from pydub import AudioSegment

# Charger le modèle Whisper une seule fois
WHISPER_MODEL = whisper.load_model("medium")  # ou "base", "small"

def convert_to_wav_if_needed(audio_path):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext != ".wav":
        audio = AudioSegment.from_file(audio_path)
        wav_path = os.path.splitext(audio_path)[0] + "_converted.wav"
        
        # Export en WAV PCM 16-bit mono, 16kHz
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(wav_path, format="wav")
        
        print(f"[i] Audio converti en WAV (PCM 16kHz mono 16-bit) : {wav_path}")
        return wav_path
    return audio_path


def transcribe_audio_simple(audio_path: str):
    try:
        audio_path = convert_to_wav_if_needed(audio_path)
        print(f"[i] Transcription du fichier audio : {audio_path}")
        
        result = WHISPER_MODEL.transcribe(audio_path, language="fr")  # langue forcée
        
        text = result.get("text", "").strip()
        language = result.get("language", None)

        print(f"[🔊] Transcription Whisper : {text}")
        print(f"[🌐] Langue détectée par Whisper : {language}")

        return text, language
    except Exception as e:
        print(f"[❌] Erreur transcription : {e}")
        return "", None

'''
if __name__ == "__main__":
    test_audio = "response_1753350310.mp3"
    texte, langue = transcribe_audio_simple(test_audio)
    print(f"Texte transcrit : {texte}")
    print(f"Langue détectée : {langue}")'''

