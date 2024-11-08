import asyncio
from fastapi import FastAPI, BackgroundTasks, HTTPException, File, UploadFile, Depends
from contextlib import asynccontextmanager
from pymongo import MongoClient
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import spacy
from spacy.matcher import PhraseMatcher
import uvicorn
import os

from scripts.pdf import get_text_from_pdf_stream
from scripts.ner import extract_skills_from_text

nlp = None
skill_extractor = None
mongo_client = None


def load_nlp_model():
    global nlp, skill_extractor
    nlp = spacy.load("en_core_web_lg")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


async def get_nlp_model():
    while skill_extractor is None or nlp is None:
        await asyncio.sleep(0.1)

    return skill_extractor


async def connect_to_mongodb():
    global mongo_client
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))


def shutdown_mongodb_client():
    if mongo_client is not None:
        mongo_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(connect_to_mongodb)
    background_tasks.add_task(load_nlp_model)
    await background_tasks()
    yield
    shutdown_mongodb_client()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return { "message": "Ping Pong!" }


@app.post("/predict")
async def predict(file: UploadFile = File(...), skill_extractor: SkillExtractor = Depends(get_nlp_model)):
    VALID_MIMETYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    if file.content_type not in VALID_MIMETYPES:
        raise HTTPException(status_code=400, detail="Invalid File Type")

    text = get_text_from_pdf_stream(file)
    skills = extract_skills_from_text(skill_extractor, text)

    return {
        "skills": skills
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
