# sentiment_analysis.py
import os
import torch
import numpy as np
from transformers import pipeline
from pydub import AudioSegment
import tempfile
import logging

# Configuration des logs
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# Désactiver les logs verbeux
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("pydub").setLevel(logging.ERROR)

# Modèle Hugging Face pour l'analyse d'émotions
MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"

# Charger le modèle une seule fois
if 'classifier' not in globals():
    classifier = pipeline(
        "audio-classification", 
        model=MODEL_NAME,
        device=0 if torch.cuda.is_available() else -1
    )

# Mapping des émotions
EMOTION_MAPPING = {
    "anger": "Colère / Frustration",
    "sadness": "Tristesse / Déception",
    "happiness": "Joie / Excitation",
    "neutral": "Concentration / Engagement",
    "disgust": "Dégoût / Mépris",
    "fear": "Peur / Anxiété",
    "surprise": "Surprise / Étonnement"
}

# Dimensions émotionnelles (Modèle MSP-Podcast)
DIMENSION_MAPPING = {
    0: "Valence (Positif/Négatif)",
    1: "Activation (Calme/Énergique)",
    2: "Dominance (Soumis/Contrôlant)"
}

def convert_to_wav(input_path):
    """Convertit n'importe quel format audio en WAV 16kHz mono"""
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.export(tmp.name, format="wav")
            return tmp.name
    except Exception as e:
        logger.error(f"Erreur de conversion audio: {str(e)}")
        return None

def predict_emotion(file_path):
    """Analyse l'émotion dans un fichier audio"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Fichier audio introuvable : {file_path}")
    
    try:
        # Conversion si nécessaire
        if not file_path.lower().endswith('.wav'):
            file_path = convert_to_wav(file_path)
            if not file_path:
                return "error", 0.0, None
        
        # Analyse de l'émotion
        results = classifier(file_path, top_k=3)
        
        # Récupération de l'émotion principale
        main_emotion = results[0]['label']
        confidence = results[0]['score']
        
        # Analyse dimensionnelle
        dimensional_results = classifier(
            file_path, 
            return_dimensional=True, 
            top_k=None
        )[0]
        
        return main_emotion, confidence, dimensional_results
    except Exception as e:
        logger.error(f"Erreur d'analyse: {str(e)}")
        return "error", 0.0, None

def format_dimensional_results(dimensions):
    """Formate les résultats dimensionnels"""
    formatted = []
    for i, value in enumerate(dimensions):
        dimension_name = DIMENSION_MAPPING.get(i, f"Dimension {i}")
        try:
            # Essayer de convertir en float
            value_float = float(value)
            formatted.append(f"{dimension_name}: {value_float:.2f}")
        except (ValueError, TypeError):
            # Si ce n'est pas un nombre, on affiche tel quel
            formatted.append(f"{dimension_name}: {value}")
    return formatted

    
'''
if __name__ == "__main__":
    audio_path = "converted.wav"  # Remplacer par votre chemin
    
    # Vérification du fichier
    if not os.path.exists(audio_path):
        print("❌ Fichier audio manquant. Veuillez créer 'converted.wav'")
    else:
        emotion, confidence, dimensions = predict_emotion(audio_path)
        
        if emotion != "error":
            # Formatage des résultats
            mapped_emotion = EMOTION_MAPPING.get(emotion, "Autre / Neutre")
            confidence_pct = confidence * 100
            
            print(f"\n🎯 Émotion principale: {mapped_emotion} (Confiance: {confidence_pct:.1f}%)")
            print(f"🔍 Détection: {emotion}")
            
            if dimensions:
                print("\n📊 Dimensions émotionnelles:")
                for dim_result in format_dimensional_results(dimensions):
                    print(f"  - {dim_result}")
        else:
            print("❌ Échec de l'analyse des émotions")'''