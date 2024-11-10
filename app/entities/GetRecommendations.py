from pydantic import BaseModel, Field

class GetRecommendations(BaseModel):
    id: str = Field(..., description="ObjectId of the teacher")

