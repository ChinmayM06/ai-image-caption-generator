"""
Utils Package

Provides utility functions for image processing, caching, and analytics.
"""

from .image_processor import (
    ImageProcessor,
    ImageProcessingError,
    get_image_processor,
    validate_image,
    preprocess_image,
    generate_image_hash
)

from .cache_manager import (
    CacheManager,
    CaptionCache,
    get_cache_manager,
    get_caption_cache
)

from .analytics import (
    AnalyticsManager,
    get_analytics_manager,
    record_generation,
    get_stats,
    get_summary,
    get_display_stats
)

__all__ = [
    # Image Processing
    "ImageProcessor",
    "ImageProcessingError",
    "get_image_processor",
    "validate_image",
    "preprocess_image",
    "generate_image_hash",
    
    # Cache Management
    "CacheManager",
    "CaptionCache",
    "get_cache_manager",
    "get_caption_cache",
    
    # Analytics
    "AnalyticsManager",
    "get_analytics_manager",
    "record_generation",
    "get_stats",
    "get_summary",
    "get_display_stats",
]