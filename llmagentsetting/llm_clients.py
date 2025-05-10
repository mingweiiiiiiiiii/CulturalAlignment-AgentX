import os
from typing import Optional, Any, List, Dict
from dotenv import load_dotenv
import requests
import json

class OllamaClient:
    def __init__(self, model_name: str = "phi4:14b-q4_K_M", host: str = "http://localhost:11434"):
        load_dotenv(".env", override=True)
        self.model_name = model_name
        self.host = os.getenv("OLLAMA_HOST", host)
        self.generate_url = f"{self.host}/api/generate"
        self.tags_url = f"{self.host}/api/tags"
        self.pull_url = f"{self.host}/api/pull"

    def _is_model_available(self, model_name_to_check: str) -> bool:
        """Checks if a model (e.g., 'model:tag' or 'model' which implies 'model:latest') is available locally."""
        try:
            response = requests.get(self.tags_url)
            response.raise_for_status()
            data = response.json()
            available_models = [model['name'] for model in data.get('models', [])]

            target_model = model_name_to_check
            if ":" not in model_name_to_check:
                target_model = f"{model_name_to_check}:latest"
            
            if target_model in available_models:
                return True
            # Also check the original specifier if it included a tag or if it's exactly what's listed
            if model_name_to_check in available_models:
                return True
            return False
        except requests.exceptions.RequestException as e:
            print(f"Error checking model availability with Ollama at {self.tags_url}: {e}")
            return False
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Ollama /api/tags: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while checking model availability: {e}")
            return False

    def _pull_model(self, model_name_to_pull: str):
        """Pulls a model using the Ollama API and streams status."""
        print(f"Attempting to pull model: {model_name_to_pull}...")
        payload = {"name": model_name_to_pull, "stream": True}
        try:
            with requests.post(self.pull_url, json=payload, stream=True) as response:
                response.raise_for_status()
                final_status_success = False
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        try:
                            chunk = json.loads(decoded_line)
                            status = chunk.get("status", "")
                            
                            message_detail = status
                            if "total" in chunk and "completed" in chunk and chunk["total"] > 0:
                                progress = (chunk["completed"] / chunk["total"]) * 100
                                message_detail += f" ({progress:.2f}%)"
                            
                            print(f"  {model_name_to_pull}: {message_detail}")

                            if "error" in chunk:
                                print(f"  Error pulling model {model_name_to_pull}: {chunk['error']}")
                                return False
                            if status == "success":
                                final_status_success = True
                        except json.JSONDecodeError:
                            print(f"  Ollama (raw): {decoded_line}")

                if final_status_success:
                    print(f"Successfully pulled model: {model_name_to_pull}")
                    return True
                else:
                    # This case might be hit if the stream ends before 'success' but no explicit 'error' was seen.
                    # Or if the last status was success but loop exited.
                    # If final_status_success is true, it means success was seen.
                    if not final_status_success:
                         print(f"Pull stream for {model_name_to_pull} ended without a clear success status. Please check Ollama logs.")
                    return final_status_success # Return based on whether success was seen
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama to pull model {model_name_to_pull} at {self.pull_url}: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while pulling model {model_name_to_pull}: {e}")
            return False

    def ensure_models_are_available(self, required_models: List[str]):
        """Checks for a list of required models and pulls them if not available."""
        print("Checking for required Ollama models...")
        all_pulled_successfully = True
        for model_name in required_models:
            print(f"Checking for model: {model_name}")
            if self._is_model_available(model_name):
                print(f"Model '{model_name}' is available.")
            else:
                print(f"Model '{model_name}' not found locally. Attempting to pull...")
                if not self._pull_model(model_name):
                    all_pulled_successfully = False
                    print(f"Failed to pull model '{model_name}'. Please check Ollama and try again.")
                # If pull was successful, it would have printed a success message.
        if all_pulled_successfully:
            print("Finished checking and pulling models. All required models should be available.")
        else:
            print("Finished checking models. Some models failed to pull. Please review the logs.")


    def generate(self, prompt: str, stream: bool = False, **kwargs) -> str:
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string.")

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            **kwargs
        }
        
        try:
            response = requests.post(self.generate_url, json=payload, stream=stream)
            response.raise_for_status()
            
            if stream:
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        json_chunk = json.loads(decoded_line)
                        full_response += json_chunk.get("response", "")
                        if json_chunk.get("done"):
                            break
                return full_response.strip()
            else:
                response_data = response.json()
                return response_data.get("response", "").strip()

        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama at {self.generate_url}: {e}")
            return f"Error: Unable to connect to Ollama - {e}"
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response from Ollama for generate: {e}")
            # print(f"Response text: {response.text}") # Be careful, response might not be fully available
            return "Error: Unable to decode JSON response from Ollama."
        except KeyError:
            return "Error: Unable to extract text from API response."
        except Exception as e:
            return f"An unexpected error occurred during generate: {e}"

if __name__ == "__main__":
    try:
        # Test model checking and pulling
        # Ensure OLLAMA_HOST is set in .env or environment if not using default
        test_client = OllamaClient() # Defaults to phi4:14b-q4_K_M for generation
        
        models_to_check = ["phi4:14b-q4_K_M", "mxbai-embed-large", "nonexistentmodel123abc"]
        print(f"\n--- Ensuring models ({', '.join(models_to_check)}) are available ---")
        test_client.ensure_models_are_available(models_to_check)

        # Test generation with the default model
        print(f"\n--- Testing generation with model: {test_client.model_name} ---")
        prompt = input(f"Enter your prompt for Ollama (using model {test_client.model_name}): ").strip()
        if prompt:
            return_text = test_client.generate(prompt)
            print("Ollama Generated Text:", return_text)
            
            print("\n--- Testing Streaming ---")
            return_text_stream = test_client.generate(prompt, stream=True)
            print("Ollama Generated Text (Streamed):", return_text_stream)
        else:
            print("No prompt entered for generation test.")
            
    except Exception as e:
        print(f"Error during OllamaClient test: {e}")
