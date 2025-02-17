import anthropic
import google.generativeai as genai
from openai import OpenAI

class ClosedModel():
    
    def __init__(self, provider: str, model_name: str, max_tokens = 20, temperature = 1.0):
        self.provider = provider
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
    
    def load_model(self, api_key):
        if self.provider == "anthropic":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "google":
            genai.configure(api_key=api_key)
            client = genai.GenerativeModel(
                self.model_name,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                ))
            self.client = client
        elif self.provider == "openai":
            self.client  = OpenAI(api_key=api_key)
        else:
            raise ValueError("Invalid provider")
        
    def generate(self, api_key: str, prompt: str) -> str:
        try:
            self.load_model(api_key)
            
            if self.provider == "google":
                
                chat = self.client.start_chat(history=[])
                response = chat.send_message(prompt)
                response = response.text
                
            elif self.provider == "openai":
                
                messages = [{"role": "user", "content": prompt}]
                
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                }

                completion = self.client.chat.completions.create(**params)
                response = completion.choices[0].message.content
                
            elif self.provider == "anthropic":
                message = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    ) 
                response =  message.content[0].text
            
            else:
                raise ValueError("Invalid provider")
        except Exception as e: # catching exception as gemini sometimes reject prompt
            print(f"Error: {e}")
            response = None
            
        return response