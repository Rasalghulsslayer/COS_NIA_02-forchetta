import os
import glob
import re
import json
from modules import schedule
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Modèle Pydantic pour le routeur
class FileRequest(BaseModel):
    veut_telecharger: bool = Field(description="Vrai si demande document physique.")
    nom_fichier_cible: str = Field(description="Nom exact du fichier PDF.")
    raisonnement: str = Field(description="Pourquoi ce fichier.")

def get_relevant_files(prompt, pdf_folder_path):
    all_pdfs = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
    if not prompt or not all_pdfs: return all_pdfs, True 

    mots_vides = ["le", "la", "les", "de", "du", "des", "un", "une", "fichier", "cours", "pdf", "sur"]
    cleaned_prompt = re.sub(r'[^\w\s]', '', prompt.lower())
    keywords = [word for word in cleaned_prompt.split() if word not in mots_vides and len(word) > 2]
    
    selected_files = []
    for pdf_path in all_pdfs:
        filename = os.path.basename(pdf_path).lower()
        if any(kw in filename for kw in keywords):
            selected_files.append(pdf_path)
            
    if selected_files: return list(set(selected_files)), False
    return all_pdfs, True

def smart_file_router(user_prompt, folder_path):
    all_pdfs = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*.pdf"))]
    if not all_pdfs: return None

    llm = Ollama(model="deepseek-r1:8b", temperature=0)
    
    router_prompt = (
        "Tu es un bibliothécaire intelligent. "
        "Voici les fichiers : " + json.dumps(all_pdfs) + "\n"
        "Si l'utilisateur veut un fichier spécifique, renvoie son nom exact en JSON strict."
    )
    
    parser = JsonOutputParser(pydantic_object=FileRequest)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", router_prompt),
        ("human", "{input}\n{format_instructions}")
    ])
    
    chain = prompt_template | llm | parser
    
    try:
        result = chain.invoke({
            "input": user_prompt,
            "format_instructions": parser.get_format_instructions()
        })
        if result["veut_telecharger"] and result["nom_fichier_cible"] in all_pdfs:
            return os.path.join(folder_path, result["nom_fichier_cible"])
    except Exception as e:
        print(f"Router Error: {e}")
    return None