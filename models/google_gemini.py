from .base_model import BaseModel
from google import genai

class GoogleGeminiModel(BaseModel):
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)

    def invoke(self, messages, multimodel_input=[]):
        
        file = []
        for video in multimodel_input:
            file.append(self.client.files.upload(file=video))
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=file + messages,
        )
        return response.text

