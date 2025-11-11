"""
Style Model Module

Handles caption styling using Groq API with fallback mechanisms.
Applies different writing styles to generated captions.
"""

import time
from typing import Optional
from groq import Groq
import requests

from config import groq_config, style_config


class StyleModelError(Exception):
    """Custom exception for style model errors"""
    pass


class StyleModel:
    """
    Caption styling using Groq LLM API
    
    Features:
    - Multiple style options
    - Automatic retry logic
    - Fallback to rule-based styling
    - Rate limiting handling
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize style model
        
        Args:
            api_key: Groq API key (uses config if not provided)
        """
        self.api_key = api_key or groq_config.API_KEY
        self.model_name = groq_config.MODEL_NAME
        self.max_tokens = groq_config.MAX_TOKENS
        self.temperature = groq_config.TEMPERATURE
        self.timeout = groq_config.TIMEOUT_SECONDS
        
        # Initialize Groq client
        if self.api_key:
            try:
                self.client = Groq(
                    api_key=self.api_key
                )
                self._api_available = True
                _ = self.client.models.list()
            except Exception as e:
                print(f"Warning: Groq client initialization failed: {e}")
                print(f"Attempting alternative initialization...")
                try:
                    # Alternative: Create client without extra params
                    import groq
                    self.client = groq.Client(api_key=self.api_key)
                    self._api_available = True
                except Exception as e2:
                    print(f"Alternative initialization also failed: {e2}")
                    self.client = None
                    self._api_available = False
        else:
            print("Warning: No Groq API key provided")
            self.client = None
            self._api_available = False
            
        # Retry configuration
        self.max_retries = groq_config.MAX_RETRIES
        self.retry_delay = groq_config.RETRY_DELAY_SECONDS
    
    def style_caption(
        self,
        caption: str,
        style: str = "Professional"
    ) -> str:
        """
        Apply style to caption
        
        Args:
            caption: Original caption
            style: Style to apply
            
        Returns:
            str: Styled caption
        """
        # If "None" style or no API, return original
        if style == "None" or not self._api_available:
            if style != "None":
                # Use fallback styling if API unavailable
                return self._fallback_style(caption, style)
            return caption
        
        # Try API styling with retries
        for attempt in range(self.max_retries):
            try:
                styled_caption = self._style_with_api(caption, style)
                return styled_caption
            
            except Exception as e:
                print(f"API styling attempt {attempt + 1} failed: {e}")
                
                # If last attempt, use fallback
                if attempt == self.max_retries - 1:
                    print(f"Using fallback styling for: {style}")
                    return self._fallback_style(caption, style)
                
                # Wait before retry
                time.sleep(self.retry_delay)
        
        # Fallback if all retries failed
        return self._fallback_style(caption, style)
    
    def _style_with_api(self, caption: str, style: str) -> str:
        """
        Style caption using Groq API
        
        Args:
            caption: Original caption
            style: Style to apply
            
        Returns:
            str: Styled caption
            
        Raises:
            StyleModelError: If API call fails
        """
        if not self._api_available:
            raise StyleModelError("API not available")
        
        # Get style prompt
        style_prompt = style_config.STYLES.get(
            style,
            style_config.STYLES[style_config.DEFAULT_STYLE]
        )
        
        # Construct messages
        messages = [
            {
                "role": "system",
                "content": "You are an expert at rewriting image captions in different styles. Keep the core meaning but adapt the tone and style as requested. Be concise."
            },
            {
                "role": "user",
                "content": f"{style_prompt}\n\nOriginal caption: {caption}\n\nStyled caption:"
            }
        ]
        
        try:
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=groq_config.TOP_P,
                timeout=self.timeout
            )
            
            # Extract styled caption
            styled_caption = response.choices[0].message.content.strip()
            
            # Clean up common artifacts
            styled_caption = self._clean_response(styled_caption)
            
            return styled_caption
            
        except requests.exceptions.Timeout:
            raise StyleModelError("API request timed out")
        except requests.exceptions.RequestException as e:
            raise StyleModelError(f"API request failed: {e}")
        except Exception as e:
            raise StyleModelError(f"Unexpected error: {e}")
    
    def _fallback_style(self, caption: str, style: str) -> str:
        """
        Apply rule-based styling as fallback
        
        Args:
            caption: Original caption
            style: Style to apply
            
        Returns:
            str: Styled caption using templates
        """
        template = style_config.FALLBACK_TEMPLATES.get(
            style,
            style_config.FALLBACK_TEMPLATES["Professional"]
        )
        
        return template.format(caption=caption)
    
    def _clean_response(self, text: str) -> str:
        """
        Clean up API response
        
        Args:
            text: Raw response text
            
        Returns:
            str: Cleaned text
        """
        # Remove common prefixes
        prefixes = [
            "Styled caption:",
            "Caption:",
            "Here's the styled caption:",
            "Here is the caption:",
        ]
        
        for prefix in prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire text is quoted
        if (text.startswith('"') and text.endswith('"')) or \
           (text.startswith("'") and text.endswith("'")):
            text = text[1:-1]
        
        return text.strip()
    
    def batch_style_captions(
        self,
        captions: dict,
        style: str = "Professional"
    ) -> dict:
        """
        Style multiple captions at once
        
        Args:
            captions: Dictionary of {model_name: caption}
            style: Style to apply
            
        Returns:
            dict: Dictionary of {model_name: styled_caption}
        """
        styled_captions = {}
        
        for model_name, caption in captions.items():
            try:
                styled_caption = self.style_caption(caption, style)
                styled_captions[model_name] = styled_caption
            except Exception as e:
                print(f"Error styling {model_name} caption: {e}")
                # Use original caption on error
                styled_captions[model_name] = caption
        
        return styled_captions
    
    def is_api_available(self) -> bool:
        """Check if API is available"""
        return self._api_available
    
    def test_connection(self) -> bool:
        """
        Test API connection
        
        Returns:
            bool: True if API is working
        """
        if not self._api_available:
            return False
        
        try:
            # Simple test call
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                timeout=5
            )
            return True
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False
    
    def get_available_styles(self) -> list:
        """Get list of available styles"""
        return list(style_config.STYLES.keys())
    
    def get_info(self) -> dict:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "api_available": self._api_available,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "available_styles": self.get_available_styles()
        }


# Singleton instance
_style_model = None


def get_style_model() -> StyleModel:
    """Get singleton StyleModel instance"""
    global _style_model
    if _style_model is None:
        _style_model = StyleModel()
    return _style_model


if __name__ == "__main__":
    # Test the style model
    print("=" * 60)
    print("STYLE MODEL - TEST MODE")
    print("=" * 60)
    
    # Initialize model
    style_model = StyleModel()
    
    print(f"\n✓ Style model initialized")
    print(f"  API Available: {style_model.is_api_available()}")
    print(f"  Model: {style_model.model_name}")
    
    # Get info
    print("\nModel Info:")
    info = style_model.get_info()
    for key, value in info.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    # Test connection if API available
    if style_model.is_api_available():
        print("\nTesting API connection...")
        connection_ok = style_model.test_connection()
        print(f"  Connection: {'✓ Success' if connection_ok else '✗ Failed'}")
        
        if connection_ok:
            # Test styling
            print("\nTesting caption styling:")
            test_caption = "A cat sitting on a windowsill looking outside"
            
            for style in ["Professional", "Creative", "Social Media"]:
                print(f"\n  {style}:")
                try:
                    styled = style_model.style_caption(test_caption, style)
                    print(f"    Original: {test_caption}")
                    print(f"    Styled: {styled}")
                except Exception as e:
                    print(f"    Error: {e}")
    else:
        print("\n⚠️  API not available, testing fallback styling:")
        test_caption = "A cat sitting on a windowsill looking outside"
        
        for style in ["Professional", "Creative", "Social Media"]:
            styled = style_model.style_caption(test_caption, style)
            print(f"\n  {style}: {styled}")
    
    print("\n" + "=" * 60)
    print("✓ Style model test complete")
    print("=" * 60)