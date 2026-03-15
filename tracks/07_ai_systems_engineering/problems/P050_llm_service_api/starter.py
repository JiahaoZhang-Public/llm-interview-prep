from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str


def create_app(generator):
    raise NotImplementedError("Implement P050 in this starter.")
