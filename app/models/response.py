from typing import List

from pydantic import BaseModel


class AugmentationResponse(BaseModel):
    text: List[str]
    org_text: str
