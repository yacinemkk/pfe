import sys
import re

try:
    from PyPDF2 import PdfReader
except ImportError:
    import os
    os.system("pip install PyPDF2")
    from PyPDF2 import PdfReader

def search_pdf(pdf_path, terms):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        print(f"Extraction successful. Total characters: {len(text)}")
        
        found_terms = {}
        for term in terms:
            # Case insensitive search
            matches = re.finditer(term, text, re.IGNORECASE)
            count = sum(1 for _ in matches)
            found_terms[term] = count
            
        return found_terms
        
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_file = "docs/Manuscript anonymous.pdf"
    features_to_search = [
        "reverseFirstNonEmptyPacketSize",
        "firstNonEmptyPacketSize",
        "bytesPerPacket",
        "dataByteCount",
        "octetTotalCount",
        "packet size",
        "payload",
        "feature selection",
        "removed",
        "dropped"
    ]
    
    results = search_pdf(pdf_file, features_to_search)
    if results:
        print("\nSearch Results (occurrences):")
        for term, count in results.items():
            print(f"- {term}: {count}")
