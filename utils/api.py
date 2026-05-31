import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI, Anthropic, OpenRouter, etc.).
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

        # Determine API provider for header/payload structure
        self.provider = "openai"  # Default
        if "anthropic.com" in self.base_url:
            self.provider = "anthropic"

        self.headers = self._get_headers()

        logging.debug(f"Initialized {self.model_type} API client. Provider: {self.provider}, URL: {self.base_url}")

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.provider == "anthropic":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_anthropic_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Extract system prompt and format messages for Anthropic API."""
        system_prompt = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Anthropic requires alternating user/assistant roles
                if not user_messages or user_messages[-1]["role"] != msg["role"]:
                    user_messages.append(msg)
                else:
                    # Merge consecutive messages of same role
                    user_messages[-1]["content"] += "\n\n" + msg["content"]
        return system_prompt, user_messages

    def _extract_anthropic_content(self, data: Dict[str, Any]) -> str:
        """Extract text content from Anthropic API response."""
        if data.get("type") == "error":
            raise RuntimeError(f"Anthropic API Error: {data.get('error', {}).get('message')}")
        if isinstance(data.get("content"), list):
            text_block = next((block["text"] for block in data["content"] if block.get("type") == "text"), None)
            if text_block:
                return text_block
            return ""
        return data.get("completion", "")

    def _summarize_payload_for_logs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact, secret-safe payload summary for debug logs."""
        messages = payload.get("messages", [])
        msg_roles = [msg.get("role", "?") for msg in messages if isinstance(msg, dict)]
        msg_lengths = [
            len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict)
        ]
        summary = {
            "model": payload.get("model"),
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
            "max_completion_tokens": payload.get("max_completion_tokens"),
            "message_count": len(messages),
            "message_roles": msg_roles,
            "message_lengths": msg_lengths,
            "has_reasoning": "reasoning" in payload,
            "has_min_p": "min_p" in payload,
            "provider": self.provider,
            "url": self.base_url,
        }
        return summary

    def _extract_http_error_details(self, response: Optional[requests.Response]) -> Dict[str, Any]:
        """Extract structured diagnostics from failed HTTP response."""
        if response is None:
            return {"status_code": None, "reason": None, "error": None, "body_preview": None}

        details: Dict[str, Any] = {
            "status_code": response.status_code,
            "reason": response.reason,
            "error": None,
            "body_preview": None,
        }
        try:
            body = response.json()
            details["error"] = body.get("error", body)
        except ValueError:
            text = (response.text or "").strip()
            details["body_preview"] = text[:1200] if text else None
        return details

    def _sanitize_messages_for_logs(self, messages: Any, max_content_chars: int = 1200) -> Any:
        """Sanitize/trim message content for logs while preserving structure."""
        if not isinstance(messages, list):
            return messages
        sanitized = []
        for msg in messages:
            if not isinstance(msg, dict):
                sanitized.append(msg)
                continue
            cleaned = dict(msg)
            content = cleaned.get("content")
            if isinstance(content, str) and len(content) > max_content_chars:
                cleaned["content"] = content[:max_content_chars] + "...<truncated>"
            sanitized.append(cleaned)
        return sanitized

    def _full_payload_for_logs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return payload with large text fields trimmed for debugging."""
        payload_copy = dict(payload)
        if "messages" in payload_copy:
            payload_copy["messages"] = self._sanitize_messages_for_logs(payload_copy["messages"])
        return payload_copy

    def _response_headers_for_logs(self, response: Optional[requests.Response]) -> Dict[str, str]:
        """Extract response headers useful for debugging provider-side failures."""
        if response is None:
            return {}
        wanted = {
            "content-type",
            "x-request-id",
            "request-id",
            "cf-ray",
            "server",
            "openai-processing-ms",
            "x-ratelimit-limit-requests",
            "x-ratelimit-remaining-requests",
            "x-ratelimit-reset-requests",
        }
        selected: Dict[str, str] = {}
        for key, value in response.headers.items():
            if key.lower() in wanted:
                selected[key] = value
        return selected

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, include_seed=True, min_p = 0.1, system=None) -> str:
        """
        Generic chat-completion style call.  We allow an optional random seed block.
        """

        #max_tokens=28000

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
            response: Optional[requests.Response] = None
            payload: Dict[str, Any] = {}
            try:
                # Build payload based on provider
                if self.provider == "anthropic":
                    system_prompt, user_messages = self._prepare_anthropic_messages(messages)
                    payload = {
                        "model": model,
                        "messages": user_messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    }
                    if system_prompt:
                        payload["system"] = system_prompt

                    if model == "claude-opus-4-7":
                        del payload["temperature"] # temp not supported on claude opus 4.7

                    # Disable reasoning/thinking for Anthropic models
                    payload["thinking"] = {"type": "disabled"}
                    #if model == "claude-opus-4-5-20251101":
                    #    print('enable thinking for opus 4.5')
                    #    payload["thinking"] = {
                    #        "type": "enabled",
                    #        "budget_tokens": 4096 
                    #    }
                else:
                    # OpenAI format
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

                    if 'openrouter' in self.base_url and model == "openai/gpt-5.2":
                        print('gpt5.2 no reasoning')
                        payload["reasoning"] = {"effort": "none"}

                    if model in ['gpt-5-chat-latest']:
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if model == 'anthropic/clause-sonnet-4':
                        payload['reasoning'] = {
                            "max_tokens": 0
                        }
                    if 'qwen3' in model.lower() and model != 'qwen/qwen3-next-80b-a3b-instruct':
                        # optionally disable thinking for qwen3 models
                        print('/no_think')
                        system_msg = [{"role": "system", "content": "/no_think"}]
                        messages = system_msg + messages
                        payload['messages'] = messages

                    if model == 'anthropic/claude-sonnet-4.5':
                        payload["max_tokens"] = 16000

                    if model == "hermes4":
                        print('injecting reasoning sys prompt')
                        reasoning_sys = "You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."
                        system_msg = [{"role": "system", "content": reasoning_sys}]
                        messages = system_msg + messages
                        payload["messages"] = messages
                        payload["max_tokens"] = 12000

                    if model == "moonshotai/kimi-k2-thinking" or model == "kimi-k2-thinking":
                        payload["max_tokens"] = 32000

                    # Disable reasoning for Anthropic models via OpenRouter
                    if 'openrouter' in self.base_url and model.startswith('anthropic'):
                        payload["reasoning"] = {"max_tokens": 0}

                    # Disable reasoning for all openrouter models
                    if 'openrouter' in self.base_url:
                        payload["reasoning"] = {"enabled": False}

                    if model == 'google/gemini-3.1-pro-preview':
                        print('gemini 3.1 pro minimal reasoning')
                        payload["reasoning"] = {"effort": "minimal"}

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()
                data = response.json()

                # Extract content based on provider
                if self.provider == "anthropic":
                    content = self._extract_anthropic_content(data)
                else:
                    content = data["choices"][0]["message"]["content"]

                # strip out any <think> blocks if the model yields that
                if '<think>' in content and "</think>" in content:
                    post_think = content.find('</think>') + len("</think>")
                    content = content[post_think:]
                if '<thinking>' in content and "</thinking>" in content:
                    post_think = content.find('</thinking>') + len("</thinking>")
                    content = content[post_think:]
                if '<reasoning>' in content and "</reasoning>" in content:
                    post_think = content.find('</reasoning>') + len("</reasoning>")
                    content = content[post_think:]
                return content
            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries}")
            except requests.exceptions.HTTPError as e:
                error_response = e.response or response
                details = self._extract_http_error_details(error_response)
                payload_summary = self._summarize_payload_for_logs(payload)
                payload_full = self._full_payload_for_logs(payload)
                response_headers = self._response_headers_for_logs(error_response)
                logging.error(
                    "HTTP error on attempt %s/%s: %s",
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                logging.error("HTTP failure details: %s", json.dumps(details, ensure_ascii=True))
                logging.error("Request summary: %s", json.dumps(payload_summary, ensure_ascii=True))
                logging.error("Response headers: %s", json.dumps(response_headers, ensure_ascii=True))
                logging.error("Request payload: %s", json.dumps(payload_full, ensure_ascii=True))
                logging.error(
                    "API_FAILURE provider=%s model=%s url=%s status=%s reason=%s response=%s",
                    self.provider,
                    payload.get("model"),
                    self.base_url,
                    details.get("status_code"),
                    details.get("reason"),
                    json.dumps(details.get("error") if details.get("error") is not None else details.get("body_preview"), ensure_ascii=True),
                )
                if (error_response is not None) and error_response.status_code == 429:
                    logging.warning("Rate limit. Backing off.")
                    time.sleep(self.retry_delay * (attempt + 1))
                    continue
            except Exception as e:
                logging.error(e)
                logging.warning(f"Error on attempt {attempt+1}/{self.max_retries}: {str(e)}")

            time.sleep(self.retry_delay)

        raise RuntimeError(f"Failed to generate text after {self.max_retries} attempts")


