from pydantic import BaseModel, Field

class GetSubjectNameEmbedding(BaseModel):
    id: str = Field(..., description="ObjectId of the subject")
