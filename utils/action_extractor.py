import json
import os
import subprocess
from datetime import datetime

def build_prompt(response_text: str, lang: str) -> str:
    if lang.startswith("fr"):
        prompt = f"""
Tu es un assistant qui analyse la réponse d’un chatbot dans un jeu vidéo.
Liste des émotions possibles : joyeux, triste, en colère, calme, surpris, neutre, encourageant, curieux, anxieux, déterminé, amusé.
Liste des actions possibles : parler, courir, danser, applaudir, lever_la_main, rire, sauter, marcher, suis_moi.
Destinations possibles : forêt, village, maison, rivière, montagne.
Pour chaque action impliquant un déplacement (marcher, courir, suis_moi), ajoute la destination si mentionnée, sinon "inconnue".

Réponse chatbot :
{response_text}
#+ parole qui v etre avec action

Format JSON strict :
"""
    elif lang.startswith("en"):
        prompt = f"""
You are an assistant analyzing a chatbot response in a video game.
Possible emotions: joyful, sad, angry, calm, surprised, neutral, encouraging, curious, fearful/anxious, determined, amused.
Possible actions: speak, run, dance, applaud, raise_hand, laugh, jump, walk, follow_me.
Possible destinations: forest, village, house, river, mountain.
For movement actions (walk, run, follow_me), add destination if mentioned, else "unknown".

Chatbot response:
{response_text}

JSON format:
"""
    elif lang.startswith("ar"):
        prompt = f"""
أنت مساعد يحلل رد شات بوت في لعبة فيديو.
العواطف المحتملة: سعيد، حزين، غاضب، هادئ، متفاجئ، محايد، مشجع، فضولي، قلق، مصمم، مستمتع.
الأفعال المحتملة: يتكلم، يركض، يرقص، يصفق، يرفع_يده، يضحك، يقفز، يمشي، اتبعني.
الوجهات المحتملة: غابة، قرية، منزل، نهر، جبل.
لكل فعل حركة (يمشي، يركض، اتبعني)، أضف الوجهة إذا وردت وإلا "غير_معروف".

رد الشات بوت:
{response_text}

صيغة JSON:
"""
    else:
        prompt = f"""
You are an assistant analyzing a chatbot response in a video game.
Identify the main emotion and actions.
Chatbot response:
{response_text}

JSON format:
"""
    return prompt


def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],  
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode("utf-8").strip()


def save_to_json(data: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[✅ Fichier sauvegardé : {filename}]")


def extract_emotions_actions(response_text: str, lang: str):
    prompt = build_prompt(response_text, lang)
    llm_response = call_llm(prompt)

    try:
        data = json.loads(llm_response)
    except json.JSONDecodeError as e:
        print(f"[❌ Erreur JSON] {e}")
        print("[ℹ] Contenu brut :", llm_response)
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/emotions_actions_{lang}_{timestamp}.json"
    save_to_json(data, filename)
    return data


def main():
    while True:
        lang = input("Langue (fr/en/ar ou 'quit' pour sortir) : ").strip().lower()
        if lang == "quit":
            break

        prompt_text = input("Prompt du chatbot : ").strip()
        if not prompt_text:
            print("[⚠] Prompt vide, réessaie.")
            continue

        result = extract_emotions_actions(prompt_text, lang)
        if result:
            print("[Résultat extrait]", result)

'''
if __name__ == "__main__":
    main()'''
