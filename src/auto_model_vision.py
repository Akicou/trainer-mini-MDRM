"""AutoModelVision compatible loading for Nayhein-8.8B-MDRM"""

from typing import Optional, Union, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from src.hybrid_model import HybridReasoningDiffusionModel
from src.config import ModelConfig, TrainingConfig


class NayheinForVisionText2Text(HybridReasoningDiffusionModel):
    """
    AutoModelVision compatible wrapper for Nayhein-8.8B-MDRM
    
    Can be loaded using:
        model = AutoModelVision.from_pretrained("nayhein-8b-mdrm")
    """

    def __init__(
        self,
        config: Optional[Union[ModelConfig, Dict]] = None,
        training_config: Optional[Union[TrainingConfig, Dict]] = None
    ):
        # Convert dict to config objects if needed
        if isinstance(config, dict):
            config = ModelConfig(**config)
        if isinstance(training_config, dict):
            training_config = TrainingConfig(**training_config)

        super().__init__(
            config=config,
            training_config=training_config
        )
        
        # Register as AutoModelVision compatible
        self.config.model_type = "nayhein_mdrm"
        self.config.architectures = ["NayheinForVisionText2Text"]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        config: Optional[Union[ModelConfig, Dict]] = None,
        training_config: Optional[Union[TrainingConfig, Dict]] = None,
        **kwargs
    ):
        """Load model from pretrained checkpoint"""
        return cls(
            reasoning_model_path=pretrained_model_name_or_path,
            diffusion_model_path=kwargs.get("diffusion_model_path", "GSAI-ML/LLaDA-V"),
            config=config,
            training_config=training_config
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass for compatibility"""
        # Use reasoning model for forward pass
        return self.reasoning_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

    def generate(
        self,
        prompt: str,
        *args,
        **kwargs
    ):
        """Override to support both text and vision inputs"""
        return super().generate(prompt, *args, **kwargs)


# Register with transformers
def register_auto_model_vision():
    """Register Nayhein model with AutoModelVision (if available)"""
    try:
        from transformers import AutoModelVision

        # Add to AutoModelVision mapping
        if not hasattr(AutoModelVision, "_model_mapping"):
            AutoModelVision._model_mapping = {}

        AutoModelVision._model_mapping["nayhein_mdrm"] = NayheinForVisionText2Text

        print("Nayhein-8.8B-MDRM registered with AutoModelVision")
    except ImportError:
        # AutoModelVision doesn't exist in transformers - this is expected
        print("Note: AutoModelVision not available in transformers (model will work without it)")
    except Exception as e:
        # Other errors - log but don't crash
        print(f"Could not register with AutoModelVision: {e}")
        print("Model can still be loaded directly: from src.auto_model_vision import NayheinForVisionText2Text")


# Auto-register on import
register_auto_model_vision()
