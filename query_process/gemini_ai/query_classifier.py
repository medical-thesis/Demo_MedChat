from langchain_google_genai import ChatGoogleGenerativeAI
from api_manager import APIKeyManager

class QueryClassifier:
    def __init__(self, model: str, api_key: APIKeyManager, temperature: float = 0.5):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("API key not found. Please provide a valid API key.")       
        if not self.model:
            raise ValueError("Model not found. Please provide a valid model.")
        
    def prompt_classify(self, query: str)->str:
        """
        Define the prompt template for query classification.
        """
        prompt = f"""
        You are a medical query classifier. Classify the type of question.

        Question: {query}
        
        Categories:
        - If it is about a specific disease or condition (like 'What is glaucoma?'), respond with: disease_info
        - If it is asking for diagnosis or possible diseases from symptoms (like 'I have a headache, what could it be?'), respond with: diagnosis_query
        - If it is an incomplete query (like 'yes', 'no', 'maybe'), respond with: diagnosis_query
        - If it is an query about common information (like greetings, aking for the assistant: "What is your name?"), respond with: general
        """
        return prompt
    
    def generate_response(self, query= str) -> str:
        """
        Generate a response from the model using the provided prompt.
        """
        llm = ChatGoogleGenerativeAI(model=self.model, 
                                     api_key=self.api_key.get_api_key(),
                                     temperature=self.temperature,
                                     )
        prompt = self.prompt_classify(query=query)
        response = llm.invoke(prompt)
        return response.content