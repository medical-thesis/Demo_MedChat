from langchain_google_genai import ChatGoogleGenerativeAI
from api_manager import APIKeyManager
from history import ChatHistory
class QueryNormalization:
    def __init__(self, api_key: APIKeyManager, model: str, chathistory: ChatHistory, temperature: float = 0.5):
        self.query_type = None
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.chat_history = chathistory
        
        if not self.api_key:
            raise ValueError("API key not found. Please provide a valid API key.")     
        if not self.model:
            raise ValueError("Model not found. Please provide a valid model.")  
        
    def prompt_normalization(self, query: str)->str:
        if self.chat_history:
            context = self.chat_history.get_context_chathistory()
            prompt= f"""    
                Normalize the following medical query by making it more concise and clear.
                If the original query information is unclear or related to the conversation history, use the conversation history to normalize the query.
                Rephrase if needed and remove any irrelevant information in the original query.
                Focus on maintaining the core meaning and intent of the query.
                Avoid adding any new information or context that is not present in the original query.
                Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                
                
                Example 1:
                    Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                    Normalized Query: "Are dizziness and nausea related to anemia?"
                    
                Example 2:
                    Original query: "Can medicine A be used to treat sore throat? I see different information online."
                    Normalized Query: "Can medicine A treat sore throat?"
                    
                Example 3:
                    Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                    Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                    
                Example 4:
                    Original query: "What is Glaucoma?"
                    Normalized Query: "What is Glaucoma?"
                
                Example 5:
                    Original query: "What causes Glaucoma ?"
                    Normalized Query: "What causes Glaucoma ?"
                    
                Example 6:
                    Original query: "What diseases can headaches be a sign of?"
                    Normalized Query: "What diseases can headaches be a sign of?"
                    
                Example 7:
                    Original query: "I have frequent headaches, is it due to anemia?"
                    Normalized Query: "Is frequent headache related to anemia?"  
                    
                Example 8:
                    Conversation history:
                        - Query: What are the side effects of drug A? 
                        - Response: Drug A can cause drowsiness and dry mouth.
                        - Query: I am allergic to antihistamines, can I use drug A? 
                        - Response: Drug A belongs to the antihistamine group, so you should consult your doctor before using it.

                    Original Query: "I have allergic rhinitis, can I use medicine A?"
                    Normalized Query: "Can I use medicine A for allergic rhinitis?"
                
                Example 9:
                    Conversation history:
                        - Query: What diseases can headaches be a sign of? 
                        - Response: Headaches can be caused by many things such as stress, lack of sleep or serious illnesses such as high blood pressure.
                        - Query: I have frequent headaches, is it due to anemia? 
                        - Response: Anemia can cause headaches, but tests are needed to determine exactly.

                    Original Query: "So if a headache is accompanied by dizziness, what diseases can it be?"
                    Normalized Query: "What diseases can a headache with dizziness be?"

                Given the conversation history:
                {context}
                    
                Original query: {query}
                Normalized query:
                """
        else:
            prompt = f"""
                Normalize the following medical query by making it more concise and clear.
                Rephrase if needed and remove any irrelevant information in the original query.
                Focus on maintaining the core meaning and intent of the query.
                Avoid adding any new information or context that is not present in the original query.
                Only return the normalized query itself, without add any explanations, descriptions, or additional text.
                
                Example 1:
                    Original query: "I've been feeling dizzy and nauseous lately, is it due to anemia?"
                    Normalized Query: "Are dizziness and nausea related to anemia?"
                    
                Example 2:
                    Original query: "Can medicine A be used to treat sore throat? I see different information online."
                    Normalized Query: "Can medicine A treat sore throat?"    
                
                Example 3:
                    Original query: "Hi, I think my child might have bronchiolitis. Can you provide more information on this condition?"
                    Normalized Query: "what is bronchiolitis and its symptoms in child?" 
                
                Example 4:
                    Original query: "What is Glaucoma?"
                    Normalized Query: "What is Glaucoma?"
                
                Example 5:
                    Original query: "What causes Glaucoma ?"
                    Normalized Query: "What causes Glaucoma ?"
                    
                Example 6:
                    Original query: "What diseases can headaches be a sign of?"
                    Normalized Query: "What diseases can headaches be a sign of?"
                    
                Example 7:
                    Original query: "I have frequent headaches, is it due to anemia?"
                    Normalized Query: "Is frequent headache related to anemia?"
           
                
                Original query: {query}
                Normalized query:
                """
                
        return prompt
    
    def generate_response(self, query: str) -> str:
        """
        Generate a response from the model using the provided prompt.
        """
        llm = ChatGoogleGenerativeAI(model=self.model, 
                                     api_key=self.api_key.get_api_key(),
                                     temperature=self.temperature,
                                     )
        prompt = self.prompt_normalization(query=query)
        response = llm.invoke(prompt)
        return response.content