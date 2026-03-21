import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from config import (
    GENERATION_MODEL, MAX_NEW_TOKENS, TEMPERATURE,
    TOP_K_GENERATION, TOP_P, PROMPT_TEMPLATES,
)


class LLMGenerator:

    def __init__(self, model_name: str = GENERATION_MODEL):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def build_prompt(
        self,
        query: str,
        context_chunks: List[Dict],
        template: str = "default",
    ) -> str:
        context_parts = []
        for i, chunk in enumerate(context_chunks):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Source {i+1}: {source}]\n{text}")

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
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 0.01),
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                do_sample=temperature > 0,
                early_stopping=True,
            )

        generated_texts = []
        for output in outputs:
            text = self.tokenizer.decode(output, skip_special_tokens=True)
            generated_texts.append(text)

        input_tokens = inputs["input_ids"].shape[1]
        output_tokens = outputs.shape[1]

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
                "max_new_tokens": max_new_tokens,
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
