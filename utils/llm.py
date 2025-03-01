import os
import warnings
from dataclasses import dataclass

import huggingface_hub
from vllm import LLM, SamplingParams

from api_keys import HF_ACCESS_TOKEN

warnings.filterwarnings("ignore")

BASE_DIRECTORY = "/scratch/users/erjones/models/postprocessed_models"
SUPPORTED_MODELS = ["7B-chat", "13B-chat", "70B-chat", "7B"]


@dataclass
class Message:
    role: str  # system, user, assistant
    content: str


class ChatLLM:
    def __init__(self, model_name: str, gpus: int = 2):
        self.model_name = model_name

    def chat_into_str(self, messages: list[Message]) -> str:
        raise NotImplementedError()

    def generate(self, prompt: str, **kwargs):
        raise NotImplementedError()

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        top_p: int = 1,
        max_tokens: int | None = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        raise NotImplementedError()


class LLAMA3(ChatLLM):
    def __init__(self, model_name: str, gpus: int = 2):
        super().__init__(model_name)

        huggingface_hub.login(token=HF_ACCESS_TOKEN)

        self.model = LLM(
            model=f"meta-llama/{model_name}",
            # dtype="bfloat16",
            tensor_parallel_size=gpus,
        )
        self.tokenizer = self.model.get_tokenizer()

    def chat_into_str(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 1,
        top_p: int = 1,
        max_tokens: int | None = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        return [outputs[i].outputs[0].text for i in range(len(outputs))]

    def generate(self, prompt: str, **kwargs):
        return self.generate_batch([prompt], **kwargs)[0]


class LLAMA(ChatLLM):
    def __init__(self, model_name: str, gpus: int = 2):
        super().__init__(model_name)
        assert model_name.startswith("llama")

        llama_type = model_name[len("llama-") :]
        assert llama_type in SUPPORTED_MODELS

        if llama_type == "70B-chat":
            basedir = "/data/erjones"
        elif llama_type in ["7B-chat", "13B-chat"]:
            basedir = BASE_DIRECTORY

        self.model = LLM(
            model=os.path.join(basedir, llama_type),
            dtype="bfloat16",
            tensor_parallel_size=gpus,
        )

    def chat_into_str(self, messages: list[Message]):
        system_msg = any(m.role == "system" for m in messages)

        sys_message = ""
        if system_msg:
            assert len(messages) >= 2
            assert messages[0].role == "system"
            assert messages[1].role == "user"

            sys_message = f"<<SYS>>\n{messages[0].content}\n<</SYS>>\n\n"
        else:
            sys_message = "<<SYS>>\n<</SYS>>\n\n"

        prompt = sys_message

        for m in messages:
            if m.role == "user":
                prompt += f" [INST] {m.content} [/INST]"
            elif m.role == "assistant":
                prompt += m.content
        return prompt

    def generate_batch(
        self,
        prompts: list[str],
        temperature: float = 0.0,
        top_p: int = 1,
        max_tokens: int | None = None,
        use_tqdm: bool = False,
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        return [outputs[i].outputs[0].text for i in range(len(outputs))]

    def generate(self, prompt: str, **kwargs):
        return self.generate_batch([prompt], **kwargs)[0]


MODELNAME2PATH = {
    "mistral-7B-instruct-v0.2": "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral-8x7b-instruct-v0.1": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}


class Mistral(ChatLLM):
    def __init__(self, model_name: str, gpus: int = 2) -> None:
        super().__init__(model_name)
        if model_name not in MODELNAME2PATH:
            raise NotImplementedError

        huggingface_hub.login(token=HF_ACCESS_TOKEN)

        # self.modal_name = model_name
        self.model = LLM(
            model=MODELNAME2PATH[model_name],
            dtype="bfloat16",
            tensor_parallel_size=gpus,
            download_dir="/data/arjunpatrawala/huggingface",
        )
        self.tokenizer = self.model.get_tokenizer()

    def chat_into_str(self, messages: list[Message]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False)

    def generate_batch(
        self,
        prompts: list[str],
        temperature=0.0,
        top_p=1,
        max_tokens: int | None = None,
        use_tqdm: bool = False,
    ):
        sampling_params = SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=use_tqdm)
        return [outputs[i].outputs[0].text for i in range(len(outputs))]

    def generate(self, prompt: str, **kwargs):
        return self.generate_batch([prompt], **kwargs)[0]
