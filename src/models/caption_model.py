"""
Caption Model Module

Manages BLIP and GIT models for image caption generation.
Handles model loading, inference, and memory management.
"""

import torch
from PIL import Image
from typing import Optional, Dict, Tuple
from transformers import (
    BlipProcessor, 
    BlipForConditionalGeneration,
    AutoProcessor,
    AutoModelForCausalLM
)
import gc

from config import model_config


class CaptionModelError(Exception):
    """Custom exception for caption model errors"""
    pass


class CaptionModel:
    """
    Base class for caption generation models
    
    Provides common interface for BLIP and GIT models
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        Initialize caption model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.processor = None
        self.model = None
        self._is_loaded = False
    
    def _get_device(self, requested_device: str) -> str:
        """
        Determine available device
        
        Args:
            requested_device: Requested device (cuda/cpu)
            
        Returns:
            str: Available device
        """
        if requested_device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def load(self) -> bool:
        """
        Load model into memory
        
        Returns:
            bool: True if successful
        """
        raise NotImplementedError("Subclass must implement load()")
    
    def generate_caption(
        self, 
        image: Image.Image,
        max_length: int = 50,
        num_beams: int = 3
    ) -> str:
        """
        Generate caption for image
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
        """
        raise NotImplementedError("Subclass must implement generate_caption()")
    
    def unload(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        self._is_loaded = False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded
    
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self._is_loaded
        }


class BLIPModel(CaptionModel):
    """
    BLIP (Bootstrapping Language-Image Pre-training) model
    
    Fast and efficient model for image captioning
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize BLIP model"""
        super().__init__(model_config.BLIP_MODEL_NAME, device)
        self.max_length = model_config.BLIP_MAX_LENGTH
        self.num_beams = model_config.BLIP_NUM_BEAMS
    
    def load(self) -> bool:
        """
        Load BLIP model and processor
        
        Returns:
            bool: True if successful
        """
        try:
            print(f"Loading BLIP model on {self.device}...")
            
            # Load processor
            self.processor = BlipProcessor.from_pretrained(
                self.model_name,
                cache_dir=model_config.MODEL_CACHE_DIR
            )
            
            # Load model
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_name,
                cache_dir=model_config.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self._is_loaded = True
            print(f"✓ BLIP model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
            self._is_loaded = False
            return False
    
    def generate_caption(
        self,
        image: Image.Image,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """
        Generate caption using BLIP
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
            
        Raises:
            CaptionModelError: If generation fails
        """
        if not self._is_loaded:
            raise CaptionModelError("BLIP model not loaded")
        
        try:
            # Use default values if not provided
            max_length = max_length or self.max_length
            num_beams = num_beams or self.num_beams
            
            # Preprocess image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True
            )
            
            return caption.strip()
            
        except Exception as e:
            raise CaptionModelError(f"BLIP caption generation failed: {e}")


class GITModel(CaptionModel):
    """
    GIT (Generative Image-to-text Transformer) model
    
    More detailed and accurate captions compared to BLIP
    """
    
    def __init__(self, device: str = "cuda"):
        """Initialize GIT model"""
        super().__init__(model_config.GIT_MODEL_NAME, device)
        self.max_length = model_config.GIT_MAX_LENGTH
        self.num_beams = model_config.GIT_NUM_BEAMS
    
    def load(self) -> bool:
        """
        Load GIT model and processor
        
        Returns:
            bool: True if successful
        """
        try:
            print(f"Loading GIT model on {self.device}...")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=model_config.MODEL_CACHE_DIR
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=model_config.MODEL_CACHE_DIR,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self._is_loaded = True
            print(f"✓ GIT model loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Error loading GIT model: {e}")
            self._is_loaded = False
            return False
    
    def generate_caption(
        self,
        image: Image.Image,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None
    ) -> str:
        """
        Generate caption using GIT
        
        Args:
            image: PIL Image
            max_length: Maximum caption length
            num_beams: Number of beams for beam search
            
        Returns:
            str: Generated caption
            
        Raises:
            CaptionModelError: If generation fails
        """
        if not self._is_loaded:
            raise CaptionModelError("GIT model not loaded")
        
        try:
            # Use default values if not provided
            max_length = max_length or self.max_length
            num_beams = num_beams or self.num_beams
            
            # Preprocess image
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate caption
            with torch.no_grad():
                output_ids = self.model.generate(
                    pixel_values=inputs.pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True
                )
            
            # Decode caption
            caption = self.processor.batch_decode(
                output_ids,
                skip_special_tokens=True
            )[0]
            
            return caption.strip()
            
        except Exception as e:
            raise CaptionModelError(f"GIT caption generation failed: {e}")


class CaptionModelManager:
    """
    Manager for both BLIP and GIT models
    
    Provides unified interface and handles model lifecycle
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            device: Device to use (cuda/cpu), auto-detects if None
        """
        self.device = device or model_config.DEVICE
        
        # Initialize models
        self.blip_model = BLIPModel(self.device)
        self.git_model = GITModel(self.device)
        
        # Track which models are loaded
        self._loaded_models = set()
    
    def load_all_models(self) -> Tuple[bool, bool]:
        """
        Load both models
        
        Returns:
            Tuple[bool, bool]: (blip_success, git_success)
        """
        blip_success = self.blip_model.load()
        if blip_success:
            self._loaded_models.add("blip")
        
        git_success = self.git_model.load()
        if git_success:
            self._loaded_models.add("git")
        
        return blip_success, git_success
    
    def load_model(self, model_name: str) -> bool:
        """
        Load specific model
        
        Args:
            model_name: Model to load ("blip" or "git")
            
        Returns:
            bool: True if successful
        """
        if model_name.lower() == "blip":
            success = self.blip_model.load()
            if success:
                self._loaded_models.add("blip")
            return success
        elif model_name.lower() == "git":
            success = self.git_model.load()
            if success:
                self._loaded_models.add("git")
            return success
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def generate_captions(
        self,
        image: Image.Image
    ) -> Dict[str, str]:
        """
        Generate captions from all loaded models
        
        Args:
            image: PIL Image
            
        Returns:
            Dict[str, str]: Captions from each model
        """
        captions = {}
        
        if "blip" in self._loaded_models:
            try:
                captions["blip"] = self.blip_model.generate_caption(image)
            except Exception as e:
                captions["blip"] = f"Error: {str(e)}"
        
        if "git" in self._loaded_models:
            try:
                captions["git"] = self.git_model.generate_caption(image)
            except Exception as e:
                captions["git"] = f"Error: {str(e)}"
        
        return captions
    
    def unload_all_models(self) -> None:
        """Unload all models from memory"""
        self.blip_model.unload()
        self.git_model.unload()
        self._loaded_models.clear()
    
    def get_status(self) -> dict:
        """Get status of all models"""
        return {
            "device": self.device,
            "blip": {
                "loaded": self.blip_model.is_loaded(),
                "info": self.blip_model.get_info()
            },
            "git": {
                "loaded": self.git_model.is_loaded(),
                "info": self.git_model.get_info()
            },
            "loaded_models": list(self._loaded_models)
        }


# Singleton instance
_model_manager = None


def get_model_manager() -> CaptionModelManager:
    """Get singleton CaptionModelManager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = CaptionModelManager()
    return _model_manager


if __name__ == "__main__":
    # Test the caption models
    print("=" * 60)
    print("CAPTION MODELS - TEST MODE")
    print("=" * 60)
    
    # Initialize manager
    manager = CaptionModelManager()
    print(f"\n✓ Model manager initialized")
    print(f"  Device: {manager.device}")
    
    print("\n" + "=" * 60)
    print("Loading models (this may take a few minutes)...")
    print("=" * 60)
    
    # Load models
    blip_success, git_success = manager.load_all_models()
    
    print(f"\nBLIP: {'✓ Loaded' if blip_success else '✗ Failed'}")
    print(f"GIT: {'✓ Loaded' if git_success else '✗ Failed'}")
    
    print("\n" + "=" * 60)
    print("Model Status:")
    print("=" * 60)
    status = manager.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ Caption models test complete")
    print("=" * 60)
    print("\nTo test caption generation, provide a test image:")
    print("  from PIL import Image")
    print("  img = Image.open('your_image.jpg')")
    print("  captions = manager.generate_captions(img)")
    print("  print(captions)")