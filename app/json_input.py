from pydantic import BaseModel


class JsonInput(BaseModel):
    """
    Model declaring JSON input objects with specific attribute names, types and validations.
    """
    sentence1: str
    sentence2: str
