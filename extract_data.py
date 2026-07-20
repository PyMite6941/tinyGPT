import csv
from bs4 import BeautifulSoup
import os
from pypdf import PdfReader

class ExtractData():
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_extension = self.file_path.split('.')[-1].lower()

    def extract_text(self) -> str:
        if os.path.isdir(self.file_path):
            text = ""
            for filename in os.listdir(self.file_path):
                full_path = os.path.join(self.file_path, filename)
                self.file_extension = full_path.split('.')[-1].lower()
                if self.file_extension in ['txt', 'md']:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        text += f.read() + "\n"
                elif self.file_extension == 'pdf':
                    text += self._extract_text_from_pdf(full_path) + "\n"
                elif self.file_extension == 'html':
                    with open(full_path, 'r', encoding='utf-8') as f:
                        soup = BeautifulSoup(f, 'html.parser')
                        text += soup.get_text() + "\n"
            return text
        elif self.file_extension == 'pdf':
            return self._extract_text_from_pdf(self.file_path)
        elif self.file_extension in ['txt', 'md']:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif self.file_extension == 'csv':
            return self._extract_data_from_csv(self.file_path)
        elif self.file_extension == 'html':
            with open(self.file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                return soup.get_text()
        else:
            raise ValueError(f"Unsupported file type: {self.file_extension}")
        
    def _extract_text_from_pdf(self,path) -> str:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    def _extract_data_from_csv(self, path) -> str:
        text = ""
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                text += " ".join(row) + "\n"
        return text