from fastapi import FastAPI, BackgroundTasks, Form, HTTPException, File, UploadFile, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from pymongo import MongoClient
import bson
from pymongo.errors import DuplicateKeyError
from pymongo.synchronous.collection import Collection
from cachetools import TTLCache
from transformers import DistilBertTokenizer, DistilBertModel
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
import spacy
from spacy.matcher import PhraseMatcher
import os

from .entities import Skill, GetSubjectNameEmbedding, GetRecommendations
from .scripts.pdf import get_text_from_pdf_stream
from .scripts.ner import extract_skills_from_text
from .scripts.transformers import create_embedding
from .scripts.file import get_sha256_hash
from .scripts.similarity import cosine_similarity


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
subjects_collection = None

# Cache
cache = TTLCache(maxsize=100, ttl=120)


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
    global mongo_client
    global users_collection, skills_collection, subjects_collection
    mongo_client = MongoClient(os.getenv("MONGODB_URI"))
    users_collection = mongo_client["subject-allocator"]["users"]
    skills_collection = mongo_client["subject-allocator"]["skills"]
    subjects_collection = mongo_client["subject-allocator"]["subjects"]

    if mongo_client is not None:
        print("[*] Connected to MongoDB at {}".format(str(os.getenv("MONGODB_URI"))))


async def get_users_collection():
    while users_collection is None:
        await asyncio.sleep(0.1)

    return users_collection


async def get_skills_collection():
    while skills_collection is None:
        await asyncio.sleep(0.1)

    return skills_collection


async def get_subjects_collection():
    while subjects_collection is None:
        await asyncio.sleep(0.1)

    return subjects_collection


async def get_cache():
    return cache


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


# cross-origin resource sharing
origins = ["http://localhost", "http://localhost:8080"]
API_URL = os.getenv("API_URL") if "API_URL" in os.environ else None

if API_URL is not None:
    origins.append(API_URL)

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"])


@app.get("/")
async def root():
    return { "message": "Ping Pong!" }


@app.post("/get_recommendations")
async def get_recommendations(body: GetRecommendations,
                              cache: TTLCache = Depends(get_cache),
                              model: DistilBertModel = Depends(get_model),
                              tokenizer: DistilBertTokenizer = Depends(get_tokenizer),
                              subjects_collection: Collection = Depends(get_subjects_collection),
                              users_collection: Collection = Depends(get_users_collection),
                              skills_collection: Collection = Depends(get_skills_collection)):
    
    if not is_valid_object_id(body.id):
        raise HTTPException(status_code=400, detail="Input ID is not a valid ObjectId")

    id = bson.objectid.ObjectId(body.id)
    user = users_collection.find_one({ "_id": id })

    if user is None:
        raise HTTPException(status_code=404, detail="User Not Found")

    # skip if user has no associated skill words
    if len(user["skills"]) == 0:
        return { "results": [] }

    # check for cache entries
    cache_key = ""
    resume_hash = ""

    if "resume" in user and "hash" in user["resume"] and user["resume"]["hash"]:
        if len(user["resume"]["hash"] > 0):
            resume_hash = user["resume"]["hash"]

        cache_key = body.id + user["resume"]["hash"] if len(resume_hash) > 0 else body.id

        if len(cache_key) > 0 and cache_key in cache:
            return { "results": cache[cache_key] }

    # retrieve all subjects and their embeddings
    all_subjects = subjects_collection.find({}, { "name": 1, "embedding": 1 })

    # populate skill references
    populated_skills = []
    for skill_id in user["skills"]:
        skill = skills_collection.find_one({ "_id": skill_id })
        populated_skills.append(skill)

    user["skills"] = populated_skills

    # similarity search
    similar = []
    user_skills = " ".join(skill["name"] for skill in user["skills"])
    user_skills_embedding = create_embedding(user_skills, model, tokenizer)

    for subject in all_subjects:
        # for backwards-compatibility, calculate embedding of the subject name 
        # if the subject does not yet have an embedding associated to it
        if "embedding" not in subject:
            subject_name_embedding = create_embedding(subject["name"], model, tokenizer)
            subjects_collection.update_one(
                    { 
                        "_id": subject["_id"]
                    },
                    { 
                        "$set": { 
                            "embedding": subject_name_embedding
                        }
                    }
            )

            # refetch on embedding re-calculation
            subject = subjects_collection.find_one({ "_id": subject["_id"] }, { "name": 1, "embedding": 1 })

        if subject is None:
            raise HTTPException(status_code=500, detail="Failed to refetch Subject during embedding re-calculation")
        
        similarity = cosine_similarity(user_skills_embedding, subject["embedding"])

        # convert ObjectId to string
        subject["_id"] = str(subject["_id"])

        # remove embedding and allocations after processing
        subject.pop("embedding", None)

        similar.append({ 
                "token": subject["name"], 
                "similarity": similarity,
                "subject": subject
        })

    similar = sorted(similar, key=lambda x: x["similarity"], reverse=True)[:5]

    # store results in cache
    if len(cache_key) > 0:
        cache[cache_key] = similar
    
    return {
        "results": similar
    }


@app.post("/get_subject_name_embedding")
async def get_subject_name_embedding(body: GetSubjectNameEmbedding,
                                     model: DistilBertModel = Depends(get_model),
                                     tokenizer: DistilBertTokenizer = Depends(get_tokenizer),
                                     subjects_collection: Collection = Depends(get_subjects_collection)):

    if not is_valid_object_id(body.id):
        raise HTTPException(status_code=400, detail="Input ID is not a valid ObjectId")

    id = bson.objectid.ObjectId(body.id)
    subject = subjects_collection.find_one({ "_id": id })

    if subject is None:
        raise HTTPException(status_code=404, detail="Subject Not Found")

    embedding = create_embedding(subject["name"], model, tokenizer)

    subjects_collection.update_one(
                {
                    "_id": id
                },
                {
                    "$set": {
                        "embedding": embedding
                    }
                }
    )

    return {
        "message": "Done."
    }


@app.post("/analyze_resume")
async def analyze_resume(file: UploadFile = File(...),
                        teacher_id: str = Form(...),
                        skill_extractor: SkillExtractor = Depends(get_nlp_model),
                        model: DistilBertModel = Depends(get_model),
                        tokenizer: DistilBertTokenizer = Depends(get_tokenizer),
                        skills_collection: Collection = Depends(get_skills_collection),
                        users_collection: Collection = Depends(get_users_collection)):

    VALID_MIMETYPES = [
        "application/pdf",
    ]

    if file.content_type not in VALID_MIMETYPES:
        raise HTTPException(status_code=400, detail="Invalid File Type")

    if not is_valid_object_id(teacher_id):
        raise HTTPException(status_code=400, detail="Input ID is not a valid ObjectId")

    user_id = bson.objectid.ObjectId(teacher_id)
    user = users_collection.find_one({ "_id": user_id })

    if not user:
        raise HTTPException(status_code=404, detail="User Not Found")

    # retrieve the hash of the file for caching purposes
    file_hash = await get_sha256_hash(file)

    # skip if this is the same file that has been tracked before
    if "resume" in user and "hash" in user["resume"]:
        if user["resume"]["hash"] == file_hash:
            return JSONResponse(status_code=304, content={})

    # named entity recognition
    text = get_text_from_pdf_stream(file)

    if text is None:
        raise HTTPException(status_code=400, detail="Empty File")

    if len(text) == 0:
        raise HTTPException(status_code=400, detail="File has no text")

    skills = extract_skills_from_text(skill_extractor, text)

    if len(skills) == 0:
        raise HTTPException(status_code=400, detail="No skill words found")

    skill_ids = []

    # create vector/embedding for each detected skill word
    for skill in skills:
        try:
            skill_exists = skills_collection.find_one({ "name": skill })

            if skill_exists:
                continue

            embedding = create_embedding(skill, model, tokenizer)

            # store skill in database, along with its associated vector/embedding
            skill = Skill(name=skill, embedding=embedding)
            skill_record = skills_collection.insert_one(skill.model_dump())
            skill_ids.append(skill_record.inserted_id)
        except DuplicateKeyError:
            continue
    
    # update user
    users_collection.update_one(
            { 
                "_id": user_id 
            }, 
            { 
                "$set": { 
                    "skills": skill_ids,
                    "resume.url": "",
                    "resume.hash": file_hash
                }
            }
    )

    return {
        "skills": skills
    }

