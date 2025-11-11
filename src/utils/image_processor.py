"""
Image Processing Module

Handles image validation, preprocessing, and optimization for caption generation.
Ensures images meet model requirements while maintaining quality.
"""

import io
import hashlib
from pathlib import Path
from typing import Tuple, Union
from PIL import Image, ImageOps

from config import image_config


class ImageProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass


class ImageProcessor:
    """
    Enterprise-grade image processor for caption generation pipeline
    
    Responsibilities:
    - Validate image format and size
    - Resize and optimize images
    - Generate cache keys
    - Handle edge cases and errors gracefully
    """
    
    def __init__(self):
        """Initialize image processor with configuration"""
        self.max_size = image_config.MAX_FILE_SIZE_BYTES
        self.max_dimension = image_config.MAX_DIMENSION
        self.min_dimension = image_config.MIN_DIMENSION
        self.allowed_formats = image_config.ALLOWED_FORMATS
        self.quality = image_config.RESIZE_QUALITY
    
    def validate_image(self, image: Union[str, Path, Image.Image, bytes]) -> Tuple[bool, str]:
        """
        Validate image meets all requirements
        
        Args:
            image: Image path, PIL Image, or bytes
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Load image if path or bytes provided
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                img = image
            else:
                return False, f"Unsupported image type: {type(image)}"
            
            # Check format (handle None format from Gradio)
            # When Gradio passes PIL images with type="pil", format can be None
            if hasattr(img, 'format') and img.format is not None:
                if img.format.upper() not in [fmt.upper() for fmt in self.allowed_formats]:
                    return False, f"Unsupported format: {img.format}. Allowed: {', '.join(self.allowed_formats)}"
            else:
                # Format is None - likely from Gradio's PIL conversion
                # We'll validate by checking if it's a valid PIL image
                print(f"DEBUG: Image format is None (from Gradio), skipping format check")
            
            # Check dimensions
            width, height = img.size
            if width < self.min_dimension or height < self.min_dimension:
                return False, f"Image too small. Minimum: {self.min_dimension}x{self.min_dimension}px"
            
            if width > 10000 or height > 10000:
                return False, "Image dimensions too large (max: 10000x10000px)"
            
            # Check file size (if path provided)
            if isinstance(image, (str, Path)):
                file_size = Path(image).stat().st_size
                if file_size > self.max_size:
                    max_mb = self.max_size / (1024 * 1024)
                    actual_mb = file_size / (1024 * 1024)
                    return False, f"File too large: {actual_mb:.1f}MB (max: {max_mb}MB)"
            
            # Try to verify image integrity (skip if format is None)
            if hasattr(img, 'format') and img.format is not None:
                # Create a copy before verify (verify closes the file)
                img_copy = img.copy()
                img_copy.verify()
            
            return True, ""
            
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    def preprocess_image(
        self, 
        image: Union[str, Path, Image.Image, bytes]
    ) -> Tuple[Image.Image, dict]:
        """
        Preprocess image for model input
        
        Args:
            image: Image path, PIL Image, or bytes
            
        Returns:
            Tuple[Image.Image, dict]: (processed_image, metadata)
            
        Raises:
            ImageProcessingError: If preprocessing fails
        """
        try:
            print(f"DEBUG: Preprocessing image of type: {type(image)}")
            
            # Validate first
            is_valid, error_msg = self.validate_image(image)
            if not is_valid:
                print(f"DEBUG: Validation failed: {error_msg}")
                raise ImageProcessingError(error_msg)
            
            # Load image
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                img = image.copy()  # Don't modify original
            else:
                raise ImageProcessingError(f"Unsupported image type: {type(image)}")
            
            # Store original metadata
            original_size = img.size
            original_format = img.format if hasattr(img, 'format') else 'Unknown'
            original_mode = img.mode
            
            print(f"DEBUG: Original format: {original_format}, mode: {original_mode}, size: {original_size}")
            
            # Convert to RGB if needed (handles RGBA, grayscale, etc.)
            if img.mode != "RGB":
                if img.mode == "RGBA":
                    # Create white background for transparent images
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                    img = background
                else:
                    img = img.convert("RGB")
            
            # Auto-orient based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Resize if needed
            if max(img.size) > self.max_dimension:
                img = self._resize_image(img)
            
            # Generate metadata
            metadata = {
                "original_size": original_size,
                "original_format": original_format,
                "original_mode": original_mode,
                "processed_size": img.size,
                "processed_mode": img.mode,
                "was_resized": original_size != img.size,
                "was_converted": original_mode != img.mode
            }
            
            print(f"DEBUG: Preprocessing complete. Final size: {img.size}, mode: {img.mode}")
            
            return img, metadata
            
        except ImageProcessingError:
            raise
        except Exception as e:
            print(f"DEBUG: Exception during preprocessing: {str(e)}")
            raise ImageProcessingError(f"Preprocessing failed: {str(e)}")
    
    def _resize_image(self, img: Image.Image) -> Image.Image:
        """
        Resize image maintaining aspect ratio
        
        Args:
            img: PIL Image
            
        Returns:
            Image.Image: Resized image
        """
        width, height = img.size
        
        if image_config.MAINTAIN_ASPECT_RATIO:
            # Calculate new dimensions maintaining aspect ratio
            if width > height:
                new_width = self.max_dimension
                new_height = int((height / width) * self.max_dimension)
            else:
                new_height = self.max_dimension
                new_width = int((width / height) * self.max_dimension)
        else:
            new_width = self.max_dimension
            new_height = self.max_dimension
        
        # Use high-quality resampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
    
    def generate_image_hash(
        self, 
        image: Union[str, Path, Image.Image, bytes],
        algorithm: str = "md5"
    ) -> str:
        """
        Generate unique hash for image (for caching)
        
        Args:
            image: Image path, PIL Image, or bytes
            algorithm: Hash algorithm (md5, sha256)
            
        Returns:
            str: Hexadecimal hash string
        """
        try:
            # Convert to bytes
            if isinstance(image, (str, Path)):
                with open(image, "rb") as f:
                    image_bytes = f.read()
            elif isinstance(image, bytes):
                image_bytes = image
            elif isinstance(image, Image.Image):
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                image_bytes = buffer.getvalue()
            else:
                raise ValueError(f"Unsupported type for hashing: {type(image)}")
            
            # Generate hash
            if algorithm == "md5":
                return hashlib.md5(image_bytes).hexdigest()
            elif algorithm == "sha256":
                return hashlib.sha256(image_bytes).hexdigest()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
                
        except Exception as e:
            raise ImageProcessingError(f"Hash generation failed: {str(e)}")
    
    def image_to_bytes(self, img: Image.Image, format: str = "PNG") -> bytes:
        """
        Convert PIL Image to bytes
        
        Args:
            img: PIL Image
            format: Output format (PNG, JPEG)
            
        Returns:
            bytes: Image bytes
        """
        buffer = io.BytesIO()
        img.save(buffer, format=format, quality=self.quality)
        return buffer.getvalue()
    
    def get_image_info(self, image: Union[str, Path, Image.Image]) -> dict:
        """
        Get detailed image information
        
        Args:
            image: Image path or PIL Image
            
        Returns:
            dict: Image information
        """
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
                file_size = Path(image).stat().st_size
            elif isinstance(image, Image.Image):
                img = image
                file_size = len(self.image_to_bytes(img))
            else:
                raise ValueError(f"Unsupported type: {type(image)}")
            
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.size[0],
                "height": img.size[1],
                "file_size": file_size,
                "file_size_mb": file_size / (1024 * 1024),
                "aspect_ratio": img.size[0] / img.size[1],
                "megapixels": (img.size[0] * img.size[1]) / 1_000_000
            }
        except Exception as e:
            raise ImageProcessingError(f"Failed to get image info: {str(e)}")


# ============================================================================
# SINGLETON INSTANCE AND CONVENIENCE FUNCTIONS
# ============================================================================

_image_processor = None


def get_image_processor() -> ImageProcessor:
    """Get singleton ImageProcessor instance"""
    global _image_processor
    if _image_processor is None:
        _image_processor = ImageProcessor()
    return _image_processor


# Convenience wrapper functions for backward compatibility
def validate_image(image: Union[str, Path, Image.Image, bytes]) -> Tuple[bool, str]:
    """
    Convenience function: Validate image using singleton processor
    
    Args:
        image: Image path, PIL Image, or bytes
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    return get_image_processor().validate_image(image)


def preprocess_image(
    image: Union[str, Path, Image.Image, bytes]
) -> Tuple[Image.Image, dict]:
    """
    Convenience function: Preprocess image using singleton processor
    
    Args:
        image: Image path, PIL Image, or bytes
        
    Returns:
        Tuple[Image.Image, dict]: (processed_image, metadata)
    """
    return get_image_processor().preprocess_image(image)


def generate_image_hash(
    image: Union[str, Path, Image.Image, bytes],
    algorithm: str = "md5"
) -> str:
    """
    Convenience function: Generate image hash using singleton processor
    
    Args:
        image: Image path, PIL Image, or bytes
        algorithm: Hash algorithm (md5, sha256)
        
    Returns:
        str: Hexadecimal hash string
    """
    return get_image_processor().generate_image_hash(image, algorithm)


if __name__ == "__main__":
    # Test the image processor
    print("=" * 60)
    print("IMAGE PROCESSOR - TEST MODE")
    print("=" * 60)
    
    processor = get_image_processor()
    print(f"✓ ImageProcessor initialized")
    print(f"  - Max file size: {processor.max_size / (1024*1024):.1f}MB")
    print(f"  - Max dimension: {processor.max_dimension}px")
    print(f"  - Allowed formats: {', '.join(processor.allowed_formats)}")
    print(f"  - Quality: {processor.quality}")
    print("=" * 60)
    print("Ready for testing with actual images")
    print("=" * 60)