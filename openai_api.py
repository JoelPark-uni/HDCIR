import os
from typing import Any, Optional

import torch
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoModelForCausalLM, AutoTokenizer

# Placeholder for backwards compatibility
openai_api_key = "<not_used_with_local_phi3>"


class _LocalPhi3Client:
    def __init__(self):
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_id = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_model_id(self) -> str:
        return os.environ.get("PHI3_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

    def _ensure_loaded(self):
        model_id = self._resolve_model_id()
        if self.model is not None and self.tokenizer is not None and self.model_id == model_id:
            return

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_obj: Any = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if isinstance(model_obj, tuple):
            model_obj = model_obj[0]
        self.model = model_obj.to(self.device)
        self.model.eval()
        self.model_id = model_id

    @torch.no_grad()
    def completion(self, prompt: str, max_new_tokens: int = 256) -> str:
        """
        Text-only inference with Phi-3-mini.
        """
        self._ensure_loaded()
        assert self.model is not None
        assert self.tokenizer is not None

        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        new_tokens = generated[:, input_ids.shape[1]:]
        text = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
        return text.strip()


_LOCAL_PHI3 = _LocalPhi3Client()


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def openai_completion(
    prompt: str,
    engine: str = "gpt-3.5-turbo",
    max_tokens: int = 700,
    temperature: float = 0,
    api_key: Optional[str] = None,
):
    """
    Drop-in replacement for the original OpenAI call.
    Uses a local Phi-3-mini model and ignores OpenAI-only parameters.
    """
    _ = engine, temperature, api_key
    capped_tokens = max(1, min(int(max_tokens), 1024))
    return _LOCAL_PHI3.completion(prompt=prompt, max_new_tokens=capped_tokens)
