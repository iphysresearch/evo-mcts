import logging
from typing import Optional
from .base import BaseClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = 'openai'


logger = logging.getLogger(__name__)

class OpenAIClient(BaseClient):

    ClientClass = OpenAI

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model, temperature)
        
        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)
        
        self.client = self.ClientClass(api_key=api_key, base_url=base_url)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1,
                            model: Optional[str] = None, api_key: Optional[str] = None, base_url: Optional[str] = None):
        if model is not None and api_key is not None and base_url is not None:
            client = OpenAI(api_key=api_key, base_url=base_url)
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=1.0, n=n, stream=False, # used for generating reflection
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
            )
        return response.choices
