from fastapi import FastAPI
from pydantic import BaseModel


class GenerateRequest(BaseModel):
    prompt: str


def create_app(generator):
    app = FastAPI()

    @app.post("/generate")
    def generate(request: GenerateRequest):
        result = generator(request.prompt)
        return {"generated_text": result}

    return app
