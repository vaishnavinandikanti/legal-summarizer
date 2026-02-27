import PyPDF2

def get_text(file):
    reader = PyPDF2.PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text
