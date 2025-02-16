from typing import List, Literal
from pydantic import BaseModel, Field

class TopicClassification(BaseModel):
    focusing: Literal[
    'web',
    'mobile(android, ios) engineering',
    'hardware & iot',
    'ai & ml & data',
    'security & network',
    'db',
    'devops & infra',
    'game',
    'product manager',
    'design',
    'etc',
    'n/a'
    ] = Field(
        description="Most relative topic of the text"
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