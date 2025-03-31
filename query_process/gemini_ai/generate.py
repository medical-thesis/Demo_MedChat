from api_manager import APIKeyManager
from langchain_google_genai import ChatGoogleGenerativeAI

class Generate:
    def __init__(self, model: str, api_key: APIKeyManager, temperature: float = 0.5):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("API key not found. Please provide a valid API key.")       
        if not self.model:
            raise ValueError("Model not found. Please provide a valid model.")
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response from the model using the provided prompt.

        :param prompt: The prompt to be used for generation.
        :return: Generated response from the model.
        """
        llm = ChatGoogleGenerativeAI(model=self.model, 
                                     api_key=self.api_key.get_api_key(),
                                     temperature=self.temperature,
                                     )
        response = llm.invoke(prompt)
        return response.content