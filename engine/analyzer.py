import spacy

# Ensure the model is loaded
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_entities(text):
    # Analyzing the first 7000 characters for meta-data
    doc = nlp(text[:7000])
    extracted = []
    seen = set()
    
    # Priority Legal Labels
    target_labels = ["ORG", "PERSON", "GPE", "DATE", "LAW"]
    
    for ent in doc.ents:
        if ent.label_ in target_labels and ent.text.strip() not in seen:
            extracted.append({"Field": ent.text, "Type": ent.label_})
            seen.add(ent.text.strip())
            
    return extracted
