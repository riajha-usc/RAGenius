import gc
import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from config import (
    GENERATION_MODEL, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_K_GENERATION, TOP_P, PROMPT_TEMPLATES,
)

MAX_INPUT_TOKENS = 384
MAX_OUTPUT_TOKENS = 128
MAX_CONTEXT_CHARS = 1500


class LLMGenerator:

    def __init__(self, model_name: str = GENERATION_MODEL):
        self.model_name = model_name
        self.device = self._select_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.zeros(1, device="mps")
                return "mps"
            except Exception:
                return "cpu"
        return "cpu"

    def _truncate_context(self, context: str) -> str:
        if len(context) <= MAX_CONTEXT_CHARS:
            return context
        return context[:MAX_CONTEXT_CHARS] + "..."

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict],
        template: str = "default",
    ) -> str:
        context_parts = []
        total_chars = 0
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            text = chunk.get("text", "")
            part = f"[Source {i+1}: {source}]\n{text}"
            if total_chars + len(part) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - total_chars
                if remaining > 50:
                    context_parts.append(part[:remaining] + "...")
                break
            context_parts.append(part)
            total_chars += len(part)

        context = "\n\n".join(context_parts)
        prompt_template = PROMPT_TEMPLATES.get(template, PROMPT_TEMPLATES["default"])
        return prompt_template.format(context=context, question=query)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int = TOP_K_GENERATION,
        top_p: float = TOP_P,
        num_return_sequences: int = 1,
    ) -> Dict:
        effective_max_tokens = min(max_new_tokens, MAX_OUTPUT_TOKENS)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_INPUT_TOKENS,
            truncation=True,
        ).to(self.device)

        input_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if temperature > 0:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=effective_max_tokens,
                    temperature=max(temperature, 0.01),
                    top_k=top_k,
                    top_p=top_p,
                    num_return_sequences=num_return_sequences,
                    do_sample=True,
                )
            else:
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=effective_max_tokens,
                    num_return_sequences=num_return_sequences,
                    do_sample=False,
                )

        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        output_tokens = outputs.shape[1]

        del inputs
        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()

        return {
            "generated_text": generated_texts[0],
            "all_sequences": generated_texts,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model": self.model_name,
            "parameters": {
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "max_new_tokens": effective_max_tokens,
            },
        }

    def generate_with_context(
        self,
        query: str,
        context_chunks: List[Dict],
        template: str = "default",
        **kwargs,
    ) -> Dict:
        prompt = self.build_prompt(query, context_chunks, template)
        result = self.generate(prompt, **kwargs)
        result["prompt"] = prompt
        result["template"] = template
        result["num_context_chunks"] = len(context_chunks)
        return result

    def get_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "device": self.device,
            "total_parameters": total_params,
            "total_parameters_millions": round(total_params / 1e6, 1),
            "vocab_size": self.tokenizer.vocab_size,
            "max_length": self.tokenizer.model_max_length,
        }