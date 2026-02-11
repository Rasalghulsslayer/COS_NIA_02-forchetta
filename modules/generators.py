import re
from gtts import gTTS
from pptx import Presentation

def clean_mermaid_code(text):
    pattern = r"```mermaid\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    if match: return match.group(1).strip()
    return text.replace("```mermaid", "").replace("```", "").strip()

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