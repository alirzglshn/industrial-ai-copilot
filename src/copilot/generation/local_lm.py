"""a thin, reusable wrapper around a local causal lm, shared across callers"""

from collections.abc import Iterator
from functools import lru_cache


class LocalCausalLM:
    def __init__(self, model_name: str) -> None:
        # imported lazily so this module does not require torch
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()

    def chat(self, system: str, user: str, max_new_tokens: int) -> str:
        """sending one system and user turn, greedy decoding for reproducibility"""
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        with self._torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # decoding only what was generated, not the echoed prompt
        completion = generated[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(completion, skip_special_tokens=True)

    def chat_stream(self, system: str, user: str, max_new_tokens: int) -> Iterator[str]:
        """like chat, yielding text pieces as they are generated"""
        import threading

        from transformers import TextIteratorStreamer

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt")

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        try:
            yield from streamer
        finally:
            thread.join()


@lru_cache(maxsize=1)
def get_local_lm(model_name: str) -> LocalCausalLM:
    """process-wide cache keyed by model name, shared across callers"""
    return LocalCausalLM(model_name)
