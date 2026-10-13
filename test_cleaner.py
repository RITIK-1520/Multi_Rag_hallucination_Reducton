import re

def clean_pdf_text(text: str) -> str:
    # Remove running header
    text = re.sub(r'(?i)\d*\s*THE BHARATIYA NYAYA SANHITA[^\n]*', ' ', text)
    # Remove footnote markers like 92 Section 100, IPC. or 110 Section 116, IPC, 1860.
    text = re.sub(r'\d{1,4}\s+Section\s+\d+[^.\n]*\.?', ' ', text)
    text = re.sub(r'\b\d{1,3}\b\s+IPC\b[^\n\.]*\.?', ' ', text)
    # Remove bracketed amendments/repeals
    text = re.sub(r'\[.*?\]', ' ', text)
    # Fix broken word splits from PDF formatting
    text = re.sub(r'\bPRIV\s+ATE\b', 'PRIVATE', text, flags=re.I)
    text = re.sub(r'\bDEF\s+ENCE\b', 'DEFENCE', text, flags=re.I)
    text = re.sub(r'\bpu\s+nished\b', 'punished', text, flags=re.I)
    text = re.sub(r'\boff\s+ence\b', 'offence', text, flags=re.I)
    text = re.sub(r'\bimpris\s+onment\b', 'imprisonment', text, flags=re.I)
    text = re.sub(r'\bpers\s+on\b', 'person', text, flags=re.I)
    text = re.sub(r'\bprop\s+erty\b', 'property', text, flags=re.I)
    # Clean whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

print("clean_pdf_text ready")
