from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document
import PyPDF2
import pandas as pd


def custom_load_pdf(path):
    return PDFLoader(path)


def custom_load_csv(path):
    return CSVLoader(path)


class PDFLoader(BaseLoader):

    def __init__(self, path):
        self.path = path

    def load(self):
        pdfFileObj = open(self.path, 'rb')
        pdfReader = PyPDF2.PdfReader(pdfFileObj)
        text = ''
        for idx in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[idx]
            text += ' ' + pageObj.extract_text()
        doc = Document(page_content=text.strip(), metadata={"source": "",
                                                            "document_name": self.path,
                                                            "customer_name": self.path})
        return [doc]


class CSVLoader(BaseLoader):

    def __init__(self, path):
        self.path = path

    def load(self):
        docs = []
        df = pd.read_csv(self.path)
        # column_names = df.iloc[1, :].to_list()
        # df.drop([0, 1], inplace=True)
        # df.columns = column_names
        df.dropna(subset=['text'], inplace=True)
        for idx, row in df.iterrows():
            text = row['text']
            metadata = {
                "source": "",
                "document_name": self.path,
                "customer_name": self.path
            }
            docs.append(Document(page_content=text.strip(), metadata=metadata))
        print(f'Ingested rows: {len(docs)}')
        return docs
