from dotenv import load_dotenv
import os
load_dotenv()

class APIKeyManager:
    """
    Class to manage API keys for different services.
    """
    def __init__(self):
        """
        Initialize the APIKeyManager and load API keys from environment variables.
        """   
        keys = os.getenv("GEMINI_API_KEYS")
        self.api_keys = [
            key.strip() for key in keys.split(",") if key.strip()
        ]
        self.current_key_index = 0
        
        if not self.api_keys:
            raise ValueError("No API keys found in the environment variable.")
    def get_api_key(self):
        """
        Get the current API key and update the index for the next call.
        """
        if not self.api_keys:
            raise ValueError("No API keys available.")
        api_key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return api_key
        
        # attempts = 0
        # while attempts < len(self.api_keys):
        #     api_key = self.api_keys[self.current_key_index]
        #     self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
        #     if api_key:
        #         return api_key
            
        #     attempts += 1
        
        # raise ValueError("all API keys are eshausted. Please check your API keys.")
    
    def reset_api_key_index(self):
        """
        Reset the API key index to the first key.
        """
        self.current_key_index = 0
        
    def add_api_key(self, api_key: str):
        """
        Add a new API key to the list of API keys.
        """
        if api_key not in self.api_keys:
            self.api_keys.append(api_key)
        else:
            raise ValueError("API key already exists.")
        