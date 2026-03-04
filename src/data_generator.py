"""Synthetic data generation with lmstudio/ollama support"""

import requests
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
from src.config import SyntheticDataConfig


@dataclass
class TrainingSample:
    """A training sample with prompt and reasoning response"""
    prompt: str
    reasoning: str
    response: str

    def to_formatted_text(self, tag_start: str = "<think>", tag_end: str = "</think>") -> str:
        """Format sample for training"""
        return f"{self.prompt}\n{tag_start}{self.reasoning}{tag_end}\n{self.response}"


class SyntheticDataGenerator:
    """Generate synthetic training data using local LLMs"""

    # Prompt templates for generating diverse queries
    PROMPT_TEMPLATES = [
        "Explain how to {task} in {technology}",
        "How would you {task} using {technology}?",
        "Create a {artifact} for {purpose}",
        "What's the best approach to {task} with {technology}?",
        "Design a {artifact} that {purpose}",
        "I need help {task} in {technology}",
        "Write code to {task} using {technology}",
        "How do I {task} with {technology}?",
    ]

    TASKS = [
        "implement authentication", "build a REST API", "create a database schema",
        "optimize performance", "handle errors", "process data",
        "implement caching", "build a web scraper", "create unit tests",
        "design a microservice", "implement rate limiting", "build a chatbot"
    ]

    TECHNOLOGIES = [
        "Python", "Node.js", "React", "Django", "FastAPI", "Express",
        "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes", "AWS"
    ]

    def __init__(self, config: SyntheticDataConfig):
        self.config = config

    def generate_prompt(self) -> str:
        """Generate a synthetic user prompt"""
        import random
        template = random.choice(self.PROMPT_TEMPLATES)
        return template.format(
            task=random.choice(self.TASKS),
            technology=random.choice(self.TECHNOLOGIES),
            artifact=random.choice(["API", "function", "class", "module", "system"]),
            purpose=random.choice(["handles authentication", "processes data efficiently", "scales well"])
        )

    def query_lmstudio(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Query lmstudio API"""
        url = f"{self.config.host}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model or self.config.model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that thinks step by step. Put your reasoning in <think></think> tags, then provide your final answer."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"LMStudio query failed: {e}")
            return None

    def query_ollama(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """Query ollama API"""
        url = f"{self.config.ollama_host}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model or self.config.model_name or "llama3.2",
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant that thinks step by step. Put your reasoning in <think></think> tags, then provide your final answer."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {"temperature": 0.7}
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        except Exception as e:
            print(f"Ollama query failed: {e}")
            return None

    def parse_response(self, full_response: str) -> tuple[str, str]:
        """Parse thinking tags from response"""
        if "<think>" in full_response and "</think>" in full_response:
            think_start = full_response.find("<think>") + len("<think>")
            think_end = full_response.find("</think>")
            reasoning = full_response[think_start:think_end].strip()
            response = full_response[think_end + len("</think>"):].strip()
            return reasoning, response
        else:
            # Fallback: treat entire response as reasoning + response
            lines = full_response.strip().split("\n", 1)
            reasoning = lines[0] if lines else full_response
            response = lines[1] if len(lines) > 1 else ""
            return reasoning, response

    def generate_sample(self, model: Optional[str] = None) -> Optional[TrainingSample]:
        """Generate a single training sample"""
        prompt = self.generate_prompt()

        if self.config.provider == "lmstudio":
            full_response = self.query_lmstudio(prompt, model)
        elif self.config.provider == "ollama":
            full_response = self.query_ollama(prompt, model)
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

        if not full_response:
            return None

        reasoning, response = self.parse_response(full_response)
        return TrainingSample(prompt=prompt, reasoning=reasoning, response=response)

    def generate_dataset(self, num_samples: Optional[int] = None, model: Optional[str] = None) -> List[TrainingSample]:
        """Generate a dataset of training samples"""
        num_samples = num_samples or self.config.num_samples
        samples = []

        print(f"Generating {num_samples} synthetic samples using {self.config.provider}...")

        for i in range(num_samples):
            sample = self.generate_sample(model)
            if sample:
                samples.append(sample)
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{num_samples} samples")

        print(f"Successfully generated {len(samples)} samples")
        return samples

    def save_dataset(self, samples: List[TrainingSample], output_path: str):
        """Save dataset to JSON file"""
        data = [
            {
                "prompt": s.prompt,
                "reasoning": s.reasoning,
                "response": s.response,
                "formatted": s.to_formatted_text()
            }
            for s in samples
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved {len(samples)} samples to {output_path}")
