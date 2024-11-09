from fastapi import FastAPI, BackgroundTasks, Form, HTTPException, File, UploadFile, Depends
from contextlib import asynccontextmanager
import asyncio
from pymongo import MongoClient
import bson
from pymongo.errors import DuplicateKeyError
from pymongo.synchronous.collection import Collection
from transformers import DistilBertTokenizer, DistilBertModel
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import spacy
from spacy.matcher import PhraseMatcher
import uvicorn
import os

from entities.Skill import Skill
from scripts.pdf import get_text_from_pdf_stream
from scripts.ner import extract_skills_from_text
from scripts.transformers import create_embedding


# DistilBERT / Transformer
model = None
tokenizer = None

# SkillNER
nlp = None
skill_extractor = None

# MongoDB
mongo_client = None
users_collection = None
skills_collection = None


def is_valid_object_id(id: str) -> bool:
    return bson.objectid.ObjectId.is_valid(id)


def load_nlp_model():
    global nlp, skill_extractor
    nlp = spacy.load("en_core_web_lg")
    skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


def load_transformer():
    global model, tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")


async def get_nlp_model():
    while skill_extractor is None or nlp is None:
        await asyncio.sleep(0.1)

    return skill_extractor


async def get_model():
    while model is None:
        await asyncio.sleep(0.1)

    return model


async def get_tokenizer():
    while tokenizer is None:
        await asyncio.sleep(0.1)

    return tokenizer


async def connect_to_mongodb():
    global mongo_client, users_collection, skills_collection
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))
    users_collection = mongo_client["subject-allocator"]["users"]
    skills_collection = mongo_client["subject-allocator"]["skills"]


async def get_users_collection():
    while users_collection is None:
        await asyncio.sleep(0.1)

    return users_collection


async def get_skills_collection():
    while skills_collection is None:
        await asyncio.sleep(0.1)

    return skills_collection


def shutdown_mongodb_client():
    if mongo_client is not None:
        mongo_client.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    background_tasks = BackgroundTasks()
    background_tasks.add_task(connect_to_mongodb)
    background_tasks.add_task(load_nlp_model)
    background_tasks.add_task(load_transformer)
    await background_tasks()
    yield
    shutdown_mongodb_client()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return { "message": "Ping Pong!" }


@app.post("/analyze_resume")
async def analyze_resume(
        file: UploadFile = File(...),
        teacher_id: str = Form(...),
        skill_extractor: SkillExtractor = Depends(get_nlp_model),
        model: DistilBertModel = Depends(get_model),
        tokenizer: DistilBertTokenizer = Depends(get_tokenizer),
        skills_collection: Collection = Depends(get_skills_collection),
        users_collection: Collection = Depends(get_users_collection)):

    VALID_MIMETYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    if file.content_type not in VALID_MIMETYPES:
        raise HTTPException(status_code=400, detail="Invalid File Type")

    if not is_valid_object_id(teacher_id):
        raise HTTPException(status_code=400, detail="Invalid ObjectId parameter")

    user_id = bson.objectid.ObjectId(teacher_id)
    user = users_collection.find_one({ "_id": user_id })

    if not user:
        raise HTTPException(status_code=404, detail="User Not Found")

    text = get_text_from_pdf_stream(file)
    skills = extract_skills_from_text(skill_extractor, text)
    skill_ids = []

    for skill in skills:
        try:
            skill_exists = skills_collection.find_one({ "name": skill })

            if skill_exists:
                continue

            embedding = create_embedding(skill, model, tokenizer)
            skill = Skill(name=skill, embedding=embedding)
            skill_record = skills_collection.insert_one(skill.model_dump())
            skill_ids.append(skill_record.inserted_id)
        except DuplicateKeyError:
            continue

    users_collection.update_one({ "_id": user_id }, { "$set": { "skills": skill_ids } })

    return {
        "skills": skills
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
