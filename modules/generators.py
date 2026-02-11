import re
from gtts import gTTS
from pptx import Presentation

def clean_mermaid_code(text):
    """
    Nettoie le code Mermaid pour éviter les erreurs de syntaxe 10.2.4.
    """
    # 1. Supprimer la réflexion de DeepSeek (<think>...</think>)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 2. Chercher le bloc de code Mermaid avec une Regex (plus précis)
    pattern = r"```mermaid\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
    else:
        # Si pas de balises, on essaie de nettoyer le texte brut
        code = text.replace("```mermaid", "").replace("```", "").strip()

    # 3. Validation et Correction de secours
    lines = code.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # On ne garde que les lignes qui ressemblent à du code Mermaid ou des commentaires
        if line and (line.startswith("graph") or "-->" in line or line.startswith("subgraph") or line.startswith("end")):
            clean_lines.append(line)
            
    final_code = "\n".join(clean_lines)

    # Si l'IA a oublié de déclarer le type de graphique, on le force
    if not final_code.startswith("graph"):
        final_code = "graph TD\n" + final_code
        
    return final_code

def generate_audio(text, lang='fr'):
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        filename = "temp_audio.mp3"
        tts.save(filename)
        return filename
    except: return None

def generate_pptx_from_json(slides_data):
    prs = Presentation()
    title_slide = prs.slides.add_slide(prs.slide_layouts[0])
    title_slide.shapes.title.text = "Synthèse IA"
    
    bullet_layout = prs.slide_layouts[1]
    for slide_content in slides_data:
        slide = prs.slides.add_slide(bullet_layout)
        slide.shapes.title.text = slide_content.get("titre", "Slide")
        tf = slide.shapes.placeholders[1].text_frame
        for point in slide_content.get("points", []):
            p = tf.add_paragraph()
            p.text = point
            p.level = 0
            
    output_file = "synthese.pptx"
    prs.save(output_file)
    return output_file