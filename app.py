from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import traceback
import time
import mimetypes
from langdetect import detect as lang_detect
from typing import List, Optional

# Modules perso
from utils.action_extractor import extract_emotions_actions
from utils.translator import translate_text, translate_documents
from utils.whisper_handler import transcribe_audio_simple
from utils.rag import build_vectorstore_from_files, query_rag
from utils.tts_handler import text_to_speech
from utils.sentiment_analysis import predict_emotion, EMOTION_MAPPING

# Ajouter la reconnaissance des types MIME pour les formats audio
mimetypes.add_type('audio/webm', '.webm')
mimetypes.add_type('audio/wav', '.wav')
mimetypes.add_type('audio/mpeg', '.mp3')
mimetypes.add_type('audio/ogg', '.ogg')
mimetypes.add_type('audio/x-m4a', '.m4a')
mimetypes.add_type('audio/flac', '.flac')

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/tts_output", StaticFiles(directory="tts_output"), name="tts_output")

# Création des dossiers nécessaires
os.makedirs("temp_uploads", exist_ok=True)
os.makedirs("translated_docs", exist_ok=True)
os.makedirs("tts_output", exist_ok=True)
os.makedirs("faiss_index", exist_ok=True)

vectorstore = None
translated_docs_path = "translated_docs"

# Charger l'index si documents existants
if os.path.exists(translated_docs_path) and any(os.scandir(translated_docs_path)):
    file_paths_for_index = [
        os.path.join(translated_docs_path, f)
        for f in os.listdir(translated_docs_path)
        if f.endswith((".txt", ".pdf", ".docx"))
    ]
    if file_paths_for_index:
        vectorstore = build_vectorstore_from_files(file_paths_for_index, "faiss_index")
        print("[INFO] Vectorstore chargé avec succès au démarrage.")
    else:
        print("[⚠] Aucun document supporté trouvé dans translated_docs. Index non chargé.")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Récupérer la liste des documents existants
    files = []
    if os.path.exists(translated_docs_path):
        files = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "files": files,
        "message": ""
    })

@app.post("/translate", response_class=HTMLResponse)
async def translate(
    request: Request,
    target_lang: str = Form(""),
    replace_all: bool = Form(False),
    delete_files: Optional[List[str]] = Form(None),
    files: List[UploadFile] = File([])
):
    # Gérer la suppression des fichiers
    if delete_files:
        for filename in delete_files:
            file_path = os.path.join(translated_docs_path, filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Erreur suppression {file_path}: {e}")

    if not files or any(file.filename == "" for file in files):
        # Juste rafraîchir la liste des fichiers après suppression
        files_list = []
        if os.path.exists(translated_docs_path):
            files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "✅ Fichiers supprimés avec succès" if delete_files else "ℹ️ Aucun nouveau fichier à traiter",
            "files": files_list
        })

    supported_extensions = (".txt", ".pdf", ".docx")
    invalid_files = [
        file.filename for file in files 
        if not file.filename.lower().endswith(supported_extensions)
    ]
    if invalid_files:
        files_list = []
        if os.path.exists(translated_docs_path):
            files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
            
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"❌ Formats non supportés: {', '.join(invalid_files)}",
            "files": files_list
        })

    temp_upload_dir = "temp_uploads"
    translated_dir = "translated_docs"
    os.makedirs(temp_upload_dir, exist_ok=True)
    os.makedirs(translated_dir, exist_ok=True)

    if replace_all:
        for filename in os.listdir(translated_dir):
            file_path = os.path.join(translated_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Erreur suppression {file_path}: {e}")

    temp_paths = []
    for uploaded_file in files:
        temp_path = os.path.join(temp_upload_dir, uploaded_file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        temp_paths.append(temp_path)

    translated_paths = []
    if target_lang and target_lang in ["fr", "en", "ar"]:
        try:
            translated_paths = translate_documents(
                temp_paths, 
                src_lang="auto", 
                tgt_lang=target_lang,
                output_dir=translated_dir
            )
        except Exception as e:
            traceback.print_exc()
            files_list = []
            if os.path.exists(translated_docs_path):
                files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
                
            return templates.TemplateResponse("index.html", {
                "request": request,
                "message": f"❌ Erreur de traduction: {str(e)}",
                "files": files_list
            })
    else:
        # Garder les fichiers dans leur langue originale
        for path in temp_paths:
            dest_path = os.path.join(translated_dir, os.path.basename(path))
            shutil.copyfile(path, dest_path)
            translated_paths.append(dest_path)

    if not translated_paths:
        files_list = []
        if os.path.exists(translated_docs_path):
            files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
            
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": "❌ Aucun document traduit ou copié.",
            "files": files_list
        })

    file_paths_for_index = [
        os.path.join(translated_dir, f)
        for f in os.listdir(translated_dir)
        if f.endswith((".txt", ".pdf", ".docx"))
    ]

    try:
        global vectorstore
        vectorstore = build_vectorstore_from_files(file_paths_for_index, "faiss_index")
    except Exception as e:
        traceback.print_exc()
        files_list = []
        if os.path.exists(translated_docs_path):
            files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
            
        return templates.TemplateResponse("index.html", {
            "request": request,
            "message": f"❌ Erreur lors de la création de l'index : {e}",
            "files": files_list
        })

    files_list = []
    if os.path.exists(translated_docs_path):
        files_list = [f for f in os.listdir(translated_docs_path) if os.path.isfile(os.path.join(translated_docs_path, f))]
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"✅ {len(translated_paths)} fichier(s) traité(s).",
        "files": files_list
    })

@app.post("/ask_micro")
async def ask_micro(
    file: UploadFile = File(...)
):
    if vectorstore is None:
        return JSONResponse({"error": "Aucun document chargé. Veuillez d'abord uploader des documents."}, status_code=400)

    # Vérification du type de fichier uniquement par extension (Windows friendly)
    try:
        allowed_extensions = ['.webm', '.wav', '.mp3', '.ogg', '.m4a', '.flac']
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            return JSONResponse({
                "error": f"Format non supporté : {file.filename}. Types supportés: {', '.join(allowed_extensions)}"
            }, status_code=400)
    except Exception as e:
        print(f"Erreur vérification extension: {e}")

    os.makedirs("temp_uploads", exist_ok=True)
    file_extension = os.path.splitext(file.filename)[1]
    if not file_extension:
        file_extension = ".webm"  # extension par défaut

    temp_path = os.path.join("temp_uploads", f"question_{int(time.time())}{file_extension}")
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # Transcription
        question_text, detected_lang = transcribe_audio_simple(temp_path)
        if not question_text:
            return JSONResponse({
                "error": "Impossible de transcrire l'audio. Assurez-vous que l'audio contient de la parole."
            }, status_code=400)

        # Déterminer la langue de réponse (basée sur la langue détectée ou par défaut)
        langVoice = detected_lang if detected_lang in ["fr", "en", "ar"] else "fr"

        # Analyse de l'émotion de l'utilisateur
        emotion_label, confidence, _ = predict_emotion(temp_path)
        if emotion_label != "error":
            user_emotion = EMOTION_MAPPING.get(emotion_label, "Neutre")
        else:
            user_emotion = "Neutre"

        # Recherche RAG en utilisant l'émotion détectée
        response_text = query_rag(
            question_text,
            vectorstore,
            user_emotion=user_emotion,
            k=5,
            score_threshold=0.7
        )

        if not response_text or "Je n'ai pas trouvé" in response_text:
            fallback_message = {
                "fr": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question.",
                "en": "I couldn't find relevant information to answer your question.",
                "ar": "لم أجد معلومات ذات صلة للإجابة على سؤالك."
            }.get(langVoice, "Je n'ai pas trouvé d'informations pertinentes.")

            audio_path = text_to_speech(fallback_message, lang=langVoice, output_dir="tts_output")
            return JSONResponse({
                "transcribed_text": question_text,
                "response": fallback_message,
                "audio_url": f"/tts_output/{os.path.basename(audio_path)}",
                "emotions_actions": None,
                "response_lang": langVoice
            })

        # Détection d'émotions et actions sur la réponse
        emotions_actions = extract_emotions_actions(response_text, langVoice)

        try:
            if lang_detect(response_text)[:2].lower() != langVoice:
                response_text = translate_text(
                    response_text,
                    src_lang=lang_detect(response_text),
                    tgt_lang=langVoice
                )
        except Exception as e:
            print(f"Erreur traduction réponse: {e}")

        # Génération audio
        audio_path = text_to_speech(response_text, lang=langVoice, output_dir="tts_output")

        return JSONResponse({
            "transcribed_text": question_text,
            "response": response_text,
            "audio_url": f"/tts_output/{os.path.basename(audio_path)}",
            "emotions_actions": emotions_actions,
            "response_lang": langVoice
        })

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": f"Erreur lors du traitement: {str(e)}"}, status_code=500)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

@app.get("/preview/{filename}")
async def preview_file(filename: str):
    file_path = os.path.join(translated_docs_path, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Fichier non trouvé")
    
    # Pour les fichiers texte, on retourne le contenu
    if filename.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return JSONResponse({"content": content, "type": "text"})
    
    # Pour les autres types, on retourne un message
    return JSONResponse({"content": f"Prévisualisation non disponible pour {filename}", "type": "other"})

@app.get("/tts_output")
async def list_audio_files():
    return {"files": [f for f in os.listdir("tts_output") if f.endswith(".mp3")]}

@app.post("/ping_micro")
async def ping_micro(request: Request):
    return {"message": "Micro client fonctionnel"}