import dotenv
dotenv.load_dotenv()

import os
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

from .base import BaseModel
from utils.utils import token_count

key = os.getenv("OPENAI_API_KEY")
api_base = os.getenv("OPENAI_API_URL")

class OpenAIBaseModel(BaseModel):
    """
    OpenAI Model interface. 

    Arguments
    ---------
    api_base : str
        URL where the model is hosted. Can be left as None for models hosted on OpenAI's
        platform. If not provided, the implementation will look at environment variables
        `OPENAI_API_BASE`
    api_version : str
        Version of the API to use. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_VERSION`. Must be
        left as None for models hosted on OpenAI's platform
    api_key : str
        Authentication token for the API. If not provided, the implementation will derive it
        from environment variables `OPENAI_API_KEY`.
    model_name : str
        Name of the model to use. If not provided, the implementation will derive it from
        environment variables `OPENAI_MODEL`.
    engine_name : str
        Alternative for `model_name`
    temperature : float
        Temperature value to use for the model. Defaults to zero for reproducibility.
    top_p : float
        Top P value to use for the model. Defaults to 0.95
    max_tokens : int
        Maximum number of tokens to pass to the model. Defaults to 800
    frequency_penalty : float
        Frequency Penalty to use for the model.
    presence_penalty : float
        Presence Penalty to use for the model.
    """

    def __init__(
        self,
        api_base=None,
        api_version=None,
        api_key=None,
        engine_name=None,
        model_name=None,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        openai_vars = self.read_openai_env_vars() 

        api_base = api_base or openai_vars["api_base"]
        api_version = api_version or openai_vars["api_version"]
        api_key = api_key or openai_vars["api_key"]
        model_name = model_name or engine_name or openai_vars["model"]

        self.openai = OpenAI(api_key=api_key)
        
        # GPT parameters
        self.model_params = {}
        self.model_params["model"] = model_name
        self.model_params["temperature"] = temperature
        self.model_params["max_tokens"] = None
        self.model_params["frequency_penalty"] = frequency_penalty
        self.model_params["presence_penalty"] = presence_penalty

    @staticmethod
    def read_openai_env_vars():
        return {
            "api_version": os.getenv("OPENAI_API_VERSION"),
            "api_base": os.getenv("OPENAI_API_BASE"),
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL"),
        }


class OpenAIModel(OpenAIBaseModel):
    def __init__(
        self,
        api_base=None,
        api_version=None,
        api_key=None,
        engine_name=None,
        model_name=None,
        temperature=0,
        frequency_penalty=0,
        presence_penalty=0,
    ):
        super().__init__(
            api_base=api_base,
            api_version=api_version,
            api_key=api_key,
            engine_name=engine_name,
            model_name=model_name,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
    
    def summarize_response(self, response):
        """Returns the first reply from the "assistant", if available"""
        if (
            "choices" in response
            and isinstance(response["choices"], list)
            and len(response["choices"]) > 0
            and "message" in response["choices"][0]
            and "content" in response["choices"][0]["message"]
            and response["choices"][0]["message"]["role"] == "assistant"
        ):
            return response["choices"][0]["message"]["content"]

        return response


    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input: list[dict]):
        """
        OpenAI API ChatCompletion implementation

        Arguments
        ---------
        processed_input : list
            Must be list of dictionaries, where each dictionary has two keys;
            "role" defines a role in the chat (e.g. "system", "user") and
            "content" defines the actual message for that turn

        Returns
        -------
        response : OpenAI API response
            Response from the openai python library

        """
        self.model_params["max_tokens"] = 4096

        # response = self.openai.chat.completions.create(
        #     messages=processed_input,
        #     **self.model_params
        # )
         
        # client = OpenAI(
        #     base_url = "https://api.nuwaapi.com/v1",
        #     api_key = "sk-q53IvJgfwidLbt6GWS1bNVGivLcWARG3dnhYrkYlgs8nbXsl"
        # )

        client = OpenAI(
            base_url = api_base,
            api_key = key
        )
        response = client.chat.completions.create(
            messages = processed_input,
            stop=None,
            stream=False,
            **self.model_params
        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens
    
class DeepSeekR1(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "deepseek-r1-250528"
        return super().prompt(processed_input)

class GPT4o(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "gpt-4o"
        return super().prompt(processed_input)
    
class GPT4(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "gpt-4-1106-preview"
        return super().prompt(processed_input)
    
class O1(OpenAIModel):
    def prompt(self, processed_input: list[dict]):
        self.model_params["model"] = "o1"
        return super().prompt(processed_input)

class GPT5(OpenAIBaseModel):
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def prompt(self, processed_input: list[dict]):
        client = OpenAI(
            base_url = api_base,
            api_key = key
        )
        response = client.chat.completions.create(
            model="gpt-5",
            messages = processed_input,
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens