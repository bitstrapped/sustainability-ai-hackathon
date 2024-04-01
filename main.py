from fastapi import FastAPI, HTTPException
from utils.qa import KoraAI
from pydantic import BaseModel
from typing import List

app = FastAPI()

kora = KoraAI("faisals-ml-playground", "us-central1")


class NamespaceData(BaseModel):
    namespace: str
    gcs: str
    folder: str


class QuestionData(BaseModel):
    namespace: str
    question: str


# Define Document model
class Document(BaseModel):
    page_content: str
    metadata: dict


# Update AnswerData
class AnswerData(BaseModel):
    query: str
    result: str
    source_documents: List[Document]


class EvaluateAnswer(BaseModel):
    namespace: str
    question: str
    expected: str
    answer: str


class EvaluateMetric(BaseModel):
    question: str
    expected: str
    answer: str
    text: str

class SummarizeText(BaseModel):
    namespace: str
    url: str
    type: str

class GenerateText(BaseModel):
    prompt: str


@app.post("/add-namespace/")
async def add_namespace(data: NamespaceData):
    try:
        kora.add_namespace(data.gcs, data.namespace, data.folder)
        return {"status": "Namespace added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/train/{namespace}")
async def train(namespace: str):
    try:
        kora.train(namespace)
        return {"status": "Training successful for namespace: " + namespace}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/get-namespace/")
async def get_namespaces():
    try:
        namespaces =  kora.get_namespaces()
        return namespaces
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/delete-namespace/{namespace}")
async def delete_namespaces(namespace: str):
    try:
        namespaces =  kora.delete_namespaces(namespace)
        return namespaces
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask/")
async def ask_question(data: QuestionData):
    try:
        answer = kora.question(data.namespace, data.question)
        return AnswerData(**answer) 
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/summarize/")
async def summarize_text(data: SummarizeText):
    try:
        summary = kora.summarize(data.namespace, data.url, data.type)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/generate-text/")
async def generate_text(data: GenerateText):
    try:
        generated = kora.generate_text(data.prompt)
        return generated
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


#feedbac
class FeedbackData(BaseModel):
    namespace: str
    question: str
    original_answer: str
    feedback_type: str 
    value: str  


@app.post("/feedback/")
async def submit_feedback(data: FeedbackData):
    try:
        if data.feedback_type == "thumbs":
            if data.value not in ["up", "down"]:
                raise ValueError("Invalid value for thumbs feedback")

        elif data.feedback_type == "score":

            if int(data.value) not in list(range(1, 11)):
                raise ValueError("Invalid score value")

        elif data.feedback_type == "rewrite":
            return {"status": "Answer rewritten successfully",
                    "answer": kora.rewrite(data.namespace, data.original_answer)}

        elif data.feedback_type == "error_category":
            valid_categories = ["misunderstanding",
                                "hallucination", "incorrect evaluation"]
            if data.value not in valid_categories:
                raise ValueError("Invalid error category")

        else:
            raise ValueError("Invalid feedback type")

        return {"status": "Feedback submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/evaluate/")
async def ask_question(data: EvaluateAnswer):
    try:
        answer = kora.evaluate_answer(data.namespace, data.question, data.expected, data.answer)
        return EvaluateMetric(**answer)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
