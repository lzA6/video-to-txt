import requests
import json
from typing import Optional, Dict, Any, Iterator
from .llm_client import LLMClient

class OllamaClient(LLMClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.generate_url = f"{self.base_url}/api/generate"

    def generate(self,
        prompt: str,
        image_path: Optional[str] = None,
        stream: bool = False,
        model: str = "llama3.2-vision",
        temperature: float = 0.2,
        num_predict: int = 256,
        context_length: Optional[int] = None) -> Iterator[str]:
        try:
            data = {
                "model": model,
                "prompt": prompt,
                "stream": True,  # Always stream from the API
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict
                }
            }
            
            if context_length is not None:
                data["options"]["num_ctx"] = context_length

            if image_path:
                data["images"] = [self.encode_image(image_path)]
                    
            response = requests.post(self.generate_url, json=data, stream=True)
            response.raise_for_status()
            
            # This function is now always a generator
            def response_generator():
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line.decode('utf-8'))
                            if 'response' in json_response:
                                yield json_response['response']
                            if json_response.get('done'):
                                break
                        except json.JSONDecodeError:
                            continue
            
            # If the user doesn't want to stream, we consume the generator here
            if not stream:
                return "".join(list(response_generator()))
            else:
                return response_generator()
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")