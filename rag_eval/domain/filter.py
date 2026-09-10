from pydantic import BaseModel


class FilterConfig(BaseModel):
    quality_threshold: str = "ACCEPT"