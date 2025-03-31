from api_manager import APIKeyManager
from langchain_google_genai import ChatGoogleGenerativeAI

class GeneralQuery:
    def __init__(self, api_key: APIKeyManager, model: str, temperature: float = 0.5):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
        if not self.api_key:
            raise ValueError("API key not found. Please provide a valid API key.")
        if not self.model:
            raise ValueError("Model not found. Please provide a valid model.")
    
    def prompt_general(self, query: str) -> str:
        """
        Define the prompt template for generating a response.
        """
        prompt = f""" 
        Please classify the question into one of the following categories and provide the exact corresponding response: 

        - If the query is related to a greeting, respond only with: "Hi, I'm a medical assistant. How can I help you today?" 
        - If the query is related to goodbyes, respond only with: "Goodbye! Have a great day!" 
        - If the query is expresses gratitude, respond only with: "You're welcome! I'm here to help." 
        - If the query is about the chatbot, respond only with: "I'm a medical assistant here to provide health advice. I am trained to be a helpful and supportive AI assistant, so feel comfortable asking me any questions." 
        - Otherwise, respond only with: "I couldn't find the relevant information." 

        Ensure that your response strictly follows the given options without any additional text. 

        Question: {query} 
        Answer: 
        """
        
        return prompt
    
    def response_general(self, query: str) -> str:
        """
        Generate a response from the model using the provided prompt.
        """
        llm = ChatGoogleGenerativeAI(model=self.model, 
                                     api_key=self.api_key.get_api_key(),
                                     temperature=self.temperature,
                                     )
        prompt = self.prompt_general(query=query)
        response = llm.invoke(prompt)
        return response.content
          