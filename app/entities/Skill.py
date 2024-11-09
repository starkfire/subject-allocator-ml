from pydantic import BaseModel, Field
from typing import List

class Skill(BaseModel):
    name: str = Field(..., description="name of the skill")
    embedding: List[float] = Field(..., description="embedding vector associated with the skill name")

