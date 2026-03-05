"""Hybrid Auto-Regressive Reasoning to Diffusion Model"""

import torch
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig
try:
    from unsloth import FastVisionModel
    HAS_UNSLOTH = True
except (ImportError, NameError, Exception) as e:
    FastVisionModel = None
    HAS_UNSLOTH = False
from src.config import ModelConfig, TrainingConfig


class ReasoningModel:
    """Qwen3.5-0.8B-Base with reasoning capabilities"""

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3.5-0.8B-Base",
        config: Optional[ModelConfig] = None,
        use_lora: bool = True,
        training_config: Optional[TrainingConfig] = None
    ):
        self.config = config or ModelConfig()
        self.model_path = model_path
        self.use_lora = use_lora
        self.training_config = training_config or TrainingConfig()
        self.unsloth_mode = bool(self.training_config.use_unsloth_low_memory)

        # Check if CUDA is available
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        dtype = torch.bfloat16 if (use_cuda and torch.cuda.is_bf16_supported()) else torch.float16
        self.tokenizer = None

        if self.unsloth_mode and HAS_UNSLOTH:
            try:
                self.model, tokenizer = FastVisionModel.from_pretrained(
                    model_path,
                    load_in_4bit=False,
                    use_gradient_checkpointing="unsloth",
                    trust_remote_code=True,
                    device_map="auto" if use_cuda else "cpu"
                )
                self.tokenizer = tokenizer
            except Exception as exc:
                print(f"Unsloth fast load failed ({exc}); falling back to standard loader.")
                self.unsloth_mode = False

        if not self.unsloth_mode:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=dtype,
                device_map="auto" if use_cuda else "cpu"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                padding_side="right"
            )

        if self.tokenizer:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Using device: {device}, dtype: {dtype}, unsloth={self.unsloth_mode}")

        # Apply LoRA if enabled
        if use_lora:
            self._setup_lora()

        self.model.eval()

    def _setup_lora(self):
        """Setup LoRA configuration for fine-tuning"""
        lora_config = LoraConfig(
            r=self.training_config.lora_r,
            lora_alpha=self.training_config.lora_alpha,
            lora_dropout=self.training_config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def format_input(self, prompt: str) -> dict:
        """Format input with reasoning tags"""
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.training_config.max_length
        ).to(self.model.device)

    def generate_reasoning(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """Generate reasoning with <think></think> tags"""
        self.model.eval()
        inputs = self.format_input(prompt)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return generated_text

    def train(
        self,
        train_dataset,
        output_dir: str = "./checkpoints/nayhein-8b",
        **training_args
    ):
        """Fine-tune the model on reasoning data"""
        use_cuda = torch.cuda.is_available()
        use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
        use_fp16 = not use_bf16

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        max_steps = training_args.get("max_steps")
        if max_steps is None and self.training_config.max_steps:
            max_steps = self.training_config.max_steps

        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=training_args.get("batch_size", self.training_config.batch_size),
            gradient_accumulation_steps=training_args.get(
                "gradient_accumulation_steps",
                self.training_config.gradient_accumulation_steps
            ),
            learning_rate=training_args.get("learning_rate", self.training_config.learning_rate),
            num_train_epochs=training_args.get("epochs", self.training_config.num_epochs),
            max_steps=max_steps if max_steps is not None else -1,
            warmup_steps=training_args.get("warmup_steps", self.training_config.warmup_steps),
            logging_steps=10,
            save_strategy="epoch",
            fp16=use_fp16,
            bf16=use_bf16,
            use_cpu=not use_cuda,
            report_to="none",
            dataset_text_field="text",
            max_length=self.training_config.max_length,
            remove_unused_columns=False,
            gradient_checkpointing=True,
            activation_offloading=True,
            optim="adamw_8bit",
            skip_memory_metrics=True,
            packing=True,
            packing_strategy="bfd",
            pad_to_multiple_of=8,
            auto_find_batch_size=self.unsloth_mode,
            dataset_kwargs={"skip_prepare_dataset": False},
        )

        trainer = SFTTrainer(
            self.model,
            sft_config,
            data_collator=data_collator,
            train_dataset=train_dataset,
            processing_class=self.tokenizer
        )

        trainer.train()
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")

    def save(self, output_dir: str):
        """Save model and tokenizer"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

    def load(self, model_path: str):
        """Load fine-tuned model"""
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"Model loaded from {model_path}")
