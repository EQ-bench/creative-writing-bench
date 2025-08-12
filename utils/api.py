import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Mimics eqbench usage: we have 'test' vs 'judge' model_type references.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logging.debug(f"Initialized {self.model_type} API client with URL: {self.base_url}")

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, include_seed=True, min_p = 0.1, system=None) -> str:
        """
        Generic chat-completion style call.  We allow an optional random seed block.
        """
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        # Optionally add random seed block as a system message for judging tasks.
        # This allows us to get variation between iterations without using temp > 0 which compromises judging performance.
        # The reason for doing this is to understand *judging* variance from the same inputs, i.e. when
        # using --redo-judging. In most use cases you won't need to worry about this and can leave it disabled.
        if False:
            if include_seed:
                seed_lines = [
                    ''.join(random.choices(string.ascii_letters + string.digits, k=80)) for _ in range(5)
                ]
                random_seed_block = (
                    "<RANDOM SEED PLEASE IGNORE>\n" +
                    "\n".join(seed_lines) +
                    "\n</RANDOM SEED>"
                )
                messages = [{"role": "system", "content": random_seed_block}] + messages

        for attempt in range(self.max_retries):
            response = {}
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens                    
                }
                if min_p != None and model != 'o3' and self.base_url != 'https://api.openai.com/v1/chat/completions':
                    # Only use min_p for the test model (not judge).
                    # If your test model doesn't support min_p, you may need to
                    # disable this here. Alternatively you could use openrouter
                    # which will automatically omit unsupported params.
                    payload['min_p'] = min_p
                if self.base_url == 'https://api.openai.com/v1/chat/completions':
                    try:
                        del payload['min_p']
                    except:
                        pass
                
                if model == 'o3':
                    # o3 has special reqs via the openai api
                    del payload['max_tokens']
                    payload['max_completion_tokens'] = max_tokens
                    payload['temperature'] = 1
                if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                    payload['reasoning_effort']="minimal"
                    del payload['max_tokens']
                    payload['max_completion_tokens'] = max_tokens
                    payload['temperature'] = 1

                if model in ['gpt-5-chat-latest']:
                    del payload['max_tokens']
                    payload['max_completion_tokens'] = max_tokens
                    payload['temperature'] = 1
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # strip out any <think> blocks if the model yields that
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:]
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_think = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_think:]
                return content
            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.HTTPError as e:
                logging.error(e)
                if response:
                    print(response)
                if e.response.status_code == 429:
                    logging.warning("Rate limit. Backing off.")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logging.error(e)
                logging.warning(f"Error on attempt {attempt+1}/{self.max_retries}: {str(e)}")

            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")




