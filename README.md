# subject-allocator-ml

**Machine Learning API/Service for the BNAHS Subject Allocator web stack.**

## Prerequisites

The service requires the following environment variables to be defined via `.env` file, Docker, or a hosting provider's settings:

* `API_URL`
    * refers to the URL of the Core Backend API
* `MONGODB_URI`
    * URI pointing to the active MongoDB database instance

## Models

* Vector embeddings are generated using [DistilBERT](https://paperswithcode.com/paper/distilbert-a-distilled-version-of-bert).
* Named Entity Recognition for skill-related words are performed with [SkillNER](https://skillner.vercel.app/).

## Resources

### `POST /get_recommendations`

* **Content-Type**: `application/json`

Takes a User's ObjectID and identifies and returns the subjects closely related to the User's skills via [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)).

**Example Body:**

```json
{
    "id": "670aeaed2fa4fc53ac8583e0"
}
```

#### Performance Considerations

Note that this similarity search implementation does not use the built-in Atlas Vector Search feature. Cosine similarity is calculated manually.


### `POST /get_subject_name_embedding`

* **Content-Type**: `application/json`

Generates vector embeddings for an input Subject's ObjectID. The output embedding will be attached to the Subject's MongoDB document with the input ObjectID.

**Example Body:**

```json
{
    "id": "67276b678e3aec73b1ef9971"
}
```

### `POST /analyze_resume`

* **Content-Type**: `multipart/form-data`

Takes a PDF file (i.e. resume, curriculum vitae) and a Teacher/User's ID, and performs Named Entity Recognition with [SkillNER](https://skillner.vercel.app/) against the parsed PDF document. Skill word embeddings are then generated via DistilBERT.

The following form parameters must be provided:

* `file`: a PDF file (corresponding to a resume or CV)
* `teacher_id`: ObjectID of the target teacher/user

#### Performance Considerations

The `analyze_resume` endpoint can be an expensive operation as it will generate embeddings for each detected skill word. Thus, the amount of text in the resume/CV will impact the endpoint's response time. Additionally, this endpoint will also generate embeddings for skill words that do not yet have associated vector embeddings (for some reasons).

Since this service generates embeddings in a context-independent manner, to speed up the operation:

* the API service will skip skill words that have already been vectorized and stored in MongoDB.
* PDF checksums are used to skip PDFs that have already been vectorized.

