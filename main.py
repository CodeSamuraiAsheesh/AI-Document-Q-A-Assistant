import os
import shutil

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException

from pydantic import BaseModel
from dotenv import load_dotenv

from rag import add_pdf
from rag import ask_question


# -------------------------
# Load Environment Variables
# -------------------------

load_dotenv()


# -------------------------
# Create FastAPI App
# -------------------------

app = FastAPI(
    title="AI Document Q&A Assistant"
)


# -------------------------
# Create Upload Folder
# -------------------------

os.makedirs(
    "uploads",
    exist_ok=True
)


# -------------------------
# Request Model
# -------------------------

class QuestionRequest(BaseModel):

    question: str


# -------------------------
# Home Endpoint
# -------------------------

@app.get("/")
def home():

    return {
        "message":
        "AI Document Q&A Assistant is running."
    }


# -------------------------
# Upload PDF
# -------------------------

@app.post("/upload")

async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )


    file_path = os.path.join(
        "uploads",
        file.filename
    )


    with open(
        file_path,
        "wb"
    ) as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )


    try:

        chunk_count = add_pdf(
            file_path
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    return {

        "message":
        "PDF uploaded successfully.",

        "chunks_created":
        chunk_count
    }


# -------------------------
# Ask Question
# -------------------------

@app.post("/ask")

def ask(
    request: QuestionRequest
):

    try:

        answer = ask_question(
            request.question
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


    return {

        "question":
        request.question,

        "answer":
        answer
    }
