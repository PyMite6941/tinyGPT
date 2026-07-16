import os
from pypdf import PdfReader

class ExtractData():
    def __init__(self, file_path):
        self.file_path = file_path
        self.full_path = "" + file_path
        self.file_extension = self.file_path.split('.')[-1].lower()

    def extract_text(self):
        if os.path.isdir(self.file_path):
            text = ""
            for filename in os.listdir(self.file_path):
                self.full_path = os.path.join(self.file_path, filename)
                self.file_extension = self.full_path.split('.')[-1].lower()
                text += self.extract_text() + "\n"
            return text
        elif self.file_extension == 'pdf':
            return self._extract_text_from_pdf()
        elif self.file_extension in ['txt', 'md']:
            with open(self.full_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {self.file_extension}")
        
    def _extract_text_from_pdf(self):
        reader = PdfReader(self.full_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text