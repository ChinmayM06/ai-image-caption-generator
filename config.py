"""
Centralized Configuration Module

This module contains all configuration settings for the AI Image Caption Generator.
Follows the single source of truth principle for easy maintenance and deployment.
"""

import os
from pathlib import Path
from typing import Dict, List, Final
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT: Final[Path] = Path(__file__).parent
CACHE_DIR: Final[Path] = PROJECT_ROOT / "cache"
MODEL_CACHE_DIR: Final[Path] = CACHE_DIR / "models"
ANALYTICS_FILE: Final[Path] = CACHE_DIR / "analytics.json"
STATIC_DIR: Final[Path] = PROJECT_ROOT / "static"

# Create directories if they don't exist
for directory in [CACHE_DIR, MODEL_CACHE_DIR, STATIC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for caption generation models"""
    
    # BLIP Model
    BLIP_MODEL_NAME: str = "Salesforce/blip-image-captioning-base"
    BLIP_MAX_LENGTH: int = 50
    BLIP_NUM_BEAMS: int = 3
    
    # GIT Model
    GIT_MODEL_NAME: str = "microsoft/git-large-coco"
    GIT_MAX_LENGTH: int = 50
    GIT_NUM_BEAMS: int = 3
    
    # Device Configuration
    DEVICE: str = "cuda"  # Will auto-fallback to CPU if CUDA unavailable
    
    # Memory Management
    MODEL_CACHE_DIR: Path = MODEL_CACHE_DIR
    LOW_MEMORY_MODE: bool = False  # Enable for systems with <8GB GPU memory


# ============================================================================
# IMAGE PROCESSING CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ImageConfig:
    """Configuration for image validation and preprocessing"""
    
    # Size Constraints
    MAX_FILE_SIZE_MB: int = 5
    MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
    MAX_DIMENSION: int = 512  # Max width/height for model input
    MIN_DIMENSION: int = 32   # Minimum acceptable dimension
    
    # Supported Formats
    ALLOWED_FORMATS: tuple = ("JPEG", "PNG", "WEBP", "JPG")
    ALLOWED_EXTENSIONS: tuple = (".jpg", ".jpeg", ".png", ".webp")
    
    # Processing
    RESIZE_QUALITY: int = 95  # JPEG quality after resize
    MAINTAIN_ASPECT_RATIO: bool = True


# ============================================================================
# GROQ API CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class GroqConfig:
    """Configuration for Groq API styling"""
    
    # API Settings
    API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = "llama-3.1-8b-instant"
    
    # Request Parameters
    MAX_TOKENS: int = 150
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TIMEOUT_SECONDS: int = 10
    
    # Retry Logic
    MAX_RETRIES: int = 3
    RETRY_DELAY_SECONDS: float = 1.0
    
    # Rate Limiting
    REQUESTS_PER_MINUTE: int = 30


# ============================================================================
# STYLE CONFIGURATION
# ============================================================================

class StyleConfig:
    """Configuration for caption styling options"""
    
    STYLES: Final[Dict[str, str]] = {
        "None": "Keep the original caption without any modifications.",
        "Professional": "Rewrite this image caption in a professional, business-appropriate tone. Make it clear, formal, and suitable for corporate presentations or reports.",
        "Creative": "Transform this caption into a creative, artistic, and imaginative description. Use vivid language and engaging expressions.",
        "Social Media": "Rewrite this caption for social media platforms. Make it engaging, add relevant emojis, and make it shareable. Keep it under 280 characters.",
        "Technical": "Rewrite this caption with technical precision and detailed analysis. Focus on specific elements, composition, and visual characteristics."
    }
    
    DEFAULT_STYLE: Final[str] = "Professional"
    
    # Fallback templates when API fails
    FALLBACK_TEMPLATES: Final[Dict[str, str]] = {
        "Professional": "Image Description: {caption}",
        "Creative": "✨ {caption} ✨",
        "Social Media": "📸 {caption} #AI #ImageCaption",
        "Technical": "Visual Analysis: {caption}",
        "None": "{caption}"
    }


# ============================================================================
# CACHE CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class CacheConfig:
    """Configuration for caching system"""
    
    # Cache Settings
    MAX_CACHE_SIZE: int = 100  # Maximum number of cached items
    CACHE_TTL_SECONDS: int = 3600  # Time to live: 1 hour
    
    # Cache Keys
    ENABLE_CAPTION_CACHE: bool = True
    CACHE_KEY_ALGO: str = "md5"  # Hashing algorithm for cache keys


# ============================================================================
# ANALYTICS CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class AnalyticsConfig:
    """Configuration for usage analytics"""
    
    # Storage
    ANALYTICS_FILE: Path = ANALYTICS_FILE
    SAVE_INTERVAL_SECONDS: int = 30  # Auto-save every 30 seconds
    
    # Metrics to Track
    TRACK_PROCESSING_TIME: bool = True
    TRACK_STYLE_USAGE: bool = True
    TRACK_MODEL_USAGE: bool = True
    TRACK_ERROR_RATE: bool = True


# ============================================================================
# GRADIO UI CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class UIConfig:
    """Configuration for Gradio interface"""
    
    # App Metadata
    TITLE: str = "🖼️ AI Image Caption Generator"
    DESCRIPTION: str = """
    Generate professional image captions using state-of-the-art AI models.
    Upload an image and choose your preferred style - get instant captions from both BLIP and GIT models.
    """
    
    # UI Settings
    THEME: str = "soft"  # Gradio theme
    SHOW_API: bool = False
    SHOW_ERROR: bool = True
    
    # Component Settings
    IMAGE_HEIGHT: int = 400
    MAX_QUEUE_SIZE: int = 10
    
    # Example Images
    EXAMPLES_DIR: Path = STATIC_DIR / "images" / "examples"


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class LogConfig:
    """Configuration for logging"""
    
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ============================================================================
# PERFORMANCE CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class PerformanceConfig:
    """Configuration for performance optimization"""
    
    # Processing Timeouts
    MAX_PROCESSING_TIME_SECONDS: int = 30
    
    # Model Loading
    LAZY_LOAD_MODELS: bool = False  # Load models on first use vs startup
    
    # Batch Processing (future feature)
    ENABLE_BATCH_PROCESSING: bool = False
    MAX_BATCH_SIZE: int = 1


# ============================================================================
# INSTANTIATE CONFIGURATIONS
# ============================================================================

# Create singleton instances
model_config = ModelConfig()
image_config = ImageConfig()
groq_config = GroqConfig()
style_config = StyleConfig()
cache_config = CacheConfig()
analytics_config = AnalyticsConfig()
ui_config = UIConfig()
log_config = LogConfig()
performance_config = PerformanceConfig()


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config() -> tuple[bool, list[str]]:
    """
    Validate all configuration settings
    
    Returns:
        tuple: (is_valid, list_of_errors)
    """
    errors = []
    
    # Check Groq API Key
    if not groq_config.API_KEY:
        errors.append("GROQ_API_KEY not found in environment variables")
    
    # Check required directories
    required_dirs = [CACHE_DIR, MODEL_CACHE_DIR]
    for directory in required_dirs:
        if not directory.exists():
            errors.append(f"Required directory not found: {directory}")
    
    # Validate image constraints
    if image_config.MAX_DIMENSION < image_config.MIN_DIMENSION:
        errors.append("MAX_DIMENSION must be greater than MIN_DIMENSION")
    
    # Validate style options
    if not style_config.STYLES:
        errors.append("No style options configured")
    
    if style_config.DEFAULT_STYLE not in style_config.STYLES:
        errors.append(f"Default style '{style_config.DEFAULT_STYLE}' not in available styles")
    
    return len(errors) == 0, errors


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def print_config_summary() -> None:
    """Print configuration summary for debugging"""
    print("=" * 60)
    print("AI IMAGE CAPTION GENERATOR - CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Cache Directory: {CACHE_DIR}")
    print(f"Model Cache: {MODEL_CACHE_DIR}")
    print(f"\nModels:")
    print(f"  - BLIP: {model_config.BLIP_MODEL_NAME}")
    print(f"  - GIT: {model_config.GIT_MODEL_NAME}")
    print(f"  - Device: {model_config.DEVICE}")
    print(f"\nGroq API:")
    print(f"  - Model: {groq_config.MODEL_NAME}")
    print(f"  - API Key: {'✓ Configured' if groq_config.API_KEY else '✗ Missing'}")
    print(f"\nImage Processing:")
    print(f"  - Max Size: {image_config.MAX_FILE_SIZE_MB}MB")
    print(f"  - Max Dimension: {image_config.MAX_DIMENSION}px")
    print(f"  - Formats: {', '.join(image_config.ALLOWED_FORMATS)}")
    print(f"\nStyle Options: {len(style_config.STYLES)}")
    for style in style_config.STYLES.keys():
        print(f"  - {style}")
    print(f"\nCache: {cache_config.MAX_CACHE_SIZE} items")
    print(f"Analytics: {analytics_config.ANALYTICS_FILE}")
    print("=" * 60)
    
    # Validate configuration
    is_valid, errors = validate_config()
    if not is_valid:
        print("\n⚠️  CONFIGURATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print("=" * 60)
    else:
        print("\n✓ Configuration validated successfully")
        print("=" * 60)


if __name__ == "__main__":
    # Run configuration validation when executed directly
    print_config_summary()