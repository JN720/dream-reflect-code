from base_model import BaseModel
from groq import Groq

class GroqModel:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        self.client = Groq(api_key=api_key)

    def invoke(self, message, multimodel_input=None):
        if multimodel_input is not None:
            raise NotImplementedError("Multimodal input is not supported for this model")

        formatted_messages = []
        for msg in message:
            formatted_messages.append({
            "role": "user",
            "content": msg
            })
        chat_completion = self.client.chat.completions.create(
            messages=formatted_messages,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content


        