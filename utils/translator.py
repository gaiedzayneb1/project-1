import os
import logging
from typing import List, Optional
from transformers import MarianMTModel, MarianTokenizer
import docx
import fitz  # PyMuPDF
from langdetect import detect, LangDetectException

logging.basicConfig(level=logging.INFO)
_loaded_models = {}

def load_model(src_lang: str, tgt_lang: str):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    if model_name in _loaded_models:
        return _loaded_models[model_name]["tokenizer"], _loaded_models[model_name]["model"]
    
    logging.info(f"Chargement modèle {model_name}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    _loaded_models[model_name] = {"tokenizer": tokenizer, "model": model}
    return tokenizer, model

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        logging.info(f"Langue détectée automatiquement : {lang}")
        return lang
    except LangDetectException:
        logging.warning("Langue non détectée, utilisation 'en' par défaut")
        return "en"

def split_text(text: str, max_len: int = 500) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 > max_len:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_len = len(word) + 1
        else:
            current_chunk.append(word)
            current_len += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    if src_lang == "auto":
        src_lang = detect_language(text)
    
    try:
        tokenizer, model = load_model(src_lang, tgt_lang)
    except Exception as e:
        logging.error(f"Erreur chargement modèle pour {src_lang} -> {tgt_lang} : {e}")
        return ""

    chunks = split_text(text, max_len=500)
    results = []

    for chunk in chunks:
        tokens = tokenizer(chunk, return_tensors="pt", padding=True, truncation=True)
        translated = model.generate(**tokens)
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        results.append(result.strip())

    return " ".join(results)

def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".pdf":
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    else:
        raise ValueError(f"Format non supporté : {ext}")

def translate_documents(
    file_paths: List[str],
    src_lang: str,
    tgt_lang: str,
    output_dir: Optional[str] = None
) -> List[str]:
    if output_dir is None:
        output_dir = "translated_docs"
    os.makedirs(output_dir, exist_ok=True)

    translated_paths = []

    for file_path in file_paths:
        try:
            original_text = extract_text_from_file(file_path)
            if not original_text.strip():
                logging.warning(f"Fichier {file_path} vide, ignoré.")
                continue
            translated_text = translate_text(original_text, src_lang, tgt_lang)

            filename = os.path.basename(file_path)
            out_name = os.path.splitext(filename)[0] + f"_translated_{tgt_lang}.txt"
            out_path = os.path.join(output_dir, out_name)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(translated_text)

            logging.info(f"Fichier traduit sauvegardé : {out_path}")
            translated_paths.append(out_path)
        except Exception as e:
            logging.error(f"Erreur traduction {file_path} : {e}")

    return translated_paths
