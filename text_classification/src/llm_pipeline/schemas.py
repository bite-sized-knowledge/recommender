from typing import List, Literal
from pydantic import BaseModel, Field

class TopicClassification(BaseModel):
    focusing: Literal[
        'Frontend', 
        'Backend', 
        'Mobile Engineering', 
        'AI / ML', 
        'Database', 
        'Security / Network', 
        'Design', 
        'Product Manager', 
        'DevOps / Infra', 
        'Hardware / IoT', 
        'QA / Test Engineer', 
        'Culture', 
        'etc', 
        'N/A'
    ] = Field(
        description="Most relative topic of the content"
    )
    keywords: List[str] = Field(
        max_length=3,
        description="Three relative keywords extracted from the text considering the focusing topic"
    )
    content_length: int = Field(
        description="Content length of the text excluding metadata"
    )
    lang: str = Field(
        description="Language of the text"
    )