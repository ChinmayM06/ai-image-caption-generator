"""
AI Image Caption Generator - Main Application

Gradio-based web interface for generating image captions using BLIP and GIT models
with customizable styling via Groq API.
"""

import gradio as gr
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Tuple, Optional

# Import our modules
from config import ui_config, performance_config
from src.utils import (
    get_image_processor,
    get_caption_cache,
    get_analytics_manager,
    ImageProcessingError
)
from src.models import (
    get_model_manager,
    get_style_model,
    CaptionModelError,
    StyleModelError
)


class CaptionGeneratorApp:
    """
    Main application class for the caption generator
    
    Manages the Gradio interface and coordinates all components
    """
    
    def __init__(self):
        """Initialize the application"""
        print("=" * 60)
        print("🚀 INITIALIZING AI IMAGE CAPTION GENERATOR")
        print("=" * 60)
        
        # Initialize components
        self.image_processor = get_image_processor()
        self.model_manager = get_model_manager()
        self.style_model = get_style_model()
        self.cache = get_caption_cache()
        self.analytics = get_analytics_manager()
        
        print("\n✓ Components initialized")
        
        # Load models
        print("\n📦 Loading AI models (this may take a few minutes on first run)...")
        blip_success, git_success = self.model_manager.load_all_models()
        
        if not (blip_success and git_success):
            print("\n⚠️  Warning: Some models failed to load")
            print(f"   BLIP: {'✓' if blip_success else '✗'}")
            print(f"   GIT: {'✓' if git_success else '✗'}")
        else:
            print("\n✓ All models loaded successfully")
        
        # Check style model
        if self.style_model.is_api_available():
            print("✓ Groq API connected")
        else:
            print("⚠️  Groq API not available - using fallback styling")
        
        print("\n" + "=" * 60)
        print("✅ INITIALIZATION COMPLETE")
        print("=" * 60 + "\n")


    def generate_captions(
        self,
        image,  # Changed: Can be path or PIL Image
        style: str,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Generate captions for an image
        
        Args:
            image: Image path (str) or PIL Image
            style: Style to apply
            progress: Gradio progress tracker
            
        Returns:
            Tuple[str, str, str]: (blip_caption, git_caption, stats_text)
        """
        start_time = time.time()
    
        try:
            # Step 1: Validate and preprocess image
            progress(0.1, desc="Validating image...")
            
            if image is None:
                return (
                    "❌ Error: No image provided",
                    "❌ Error: No image provided",
                    "⚠️ Please upload an image"
                )
            
            # Convert to PIL Image from various formats
            try:
                if isinstance(image, str):
                    # File path
                    pil_image = Image.open(image)
                elif isinstance(image, Image.Image):
                    # Already PIL Image
                    pil_image = image
                elif hasattr(image, 'shape'):
                    # Numpy array
                    import numpy as np
                    if isinstance(image, np.ndarray):
                        pil_image = Image.fromarray(image.astype('uint8'))
                    else:
                        raise ValueError("Unsupported array type")
                else:
                    return (
                        f"❌ Error: Unsupported image type: {type(image)}",
                        f"❌ Error: Unsupported image type: {type(image)}",
                        "⚠️ Image format not supported"
                    )
            except Exception as e:
                return (
                    f"❌ Error: Cannot load image - {str(e)}",
                    f"❌ Error: Cannot load image - {str(e)}",
                    "⚠️ Image loading failed"
                )
        
            # Validate image
            is_valid, error_msg = self.image_processor.validate_image(pil_image)
            if not is_valid:
                return (
                    f"❌ Error: {error_msg}",
                    f"❌ Error: {error_msg}",
                    "⚠️ Image validation failed"
                )
        
            # Preprocess image
            progress(0.2, desc="Processing image...")
            processed_img, metadata = self.image_processor.preprocess_image(pil_image)
            
            # Generate image hash for caching
            image_hash = self.image_processor.generate_image_hash(processed_img)
            
            # Step 2: Check cache
            progress(0.3, desc="Checking cache...")
            
            blip_cached = self.cache.get_caption(image_hash, "blip", style)
            git_cached = self.cache.get_caption(image_hash, "git", style)
            
            # Step 3: Generate captions if not cached
            raw_captions = {}
        
            if blip_cached is None or git_cached is None:
                progress(0.4, desc="Generating captions...")
                raw_captions = self.model_manager.generate_captions(processed_img)
            
            # Step 4: Apply styling
            progress(0.6, desc=f"Applying {style} style...")
            
            styled_captions = {}
            
            # BLIP caption
            if blip_cached:
                styled_captions["blip"] = blip_cached
            else:
                blip_raw = raw_captions.get("blip", "Error generating caption")
                styled_captions["blip"] = self.style_model.style_caption(blip_raw, style)
                self.cache.set_caption(image_hash, "blip", style, styled_captions["blip"])
        
            # GIT caption
            if git_cached:
                styled_captions["git"] = git_cached
            else:
                git_raw = raw_captions.get("git", "Error generating caption")
                styled_captions["git"] = self.style_model.style_caption(git_raw, style)
                self.cache.set_caption(image_hash, "git", style, styled_captions["git"])
            
            # Step 5: Record analytics
            progress(0.9, desc="Finalizing...")
            
            processing_time = time.time() - start_time
            
            # Record for each model
            self.analytics.record_caption_generation("blip", style, processing_time / 2, True)
            self.analytics.record_caption_generation("git", style, processing_time / 2, True)
            
            # Get stats
            stats_text = self.analytics.get_display_stats()
            stats_text += f" | ⏱️ This generation: {processing_time:.2f}s"
            
            progress(1.0, desc="Complete!")
            
            return (
                styled_captions.get("blip", "Error"),
                styled_captions.get("git", "Error"),
                stats_text
            )
        
        except ImageProcessingError as e:
            error_msg = f"❌ Image Error: {str(e)}"
            return error_msg, error_msg, "⚠️ Image processing failed"

        except CaptionModelError as e:
            error_msg = f"❌ Model Error: {str(e)}"
            return error_msg, error_msg, "⚠️ Caption generation failed"
        
        except Exception as e:
            error_msg = f"❌ Unexpected Error: {str(e)}"
            print(f"Error in generate_captions: {e}")
            
            # Record error
            self.analytics.record_caption_generation("unknown", style, 0, False)
            
            return error_msg, error_msg, "⚠️ An error occurred"

    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title=ui_config.TITLE,
            css=self._get_custom_css()
        ) as interface:
            
            # Header
            gr.Markdown(f"# {ui_config.TITLE}")
            gr.Markdown(ui_config.DESCRIPTION)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📤 Upload Image")
                    
                    image_input = gr.Image(
                        label="Upload your image",
                        type="pil",
                        height=ui_config.IMAGE_HEIGHT
                    )
                    
                    style_dropdown = gr.Dropdown(
                        choices=self.style_model.get_available_styles(),
                        value="Professional",
                        label="🎨 Choose Caption Style",
                        info="Select how you want your caption to be styled"
                    )
                    
                    generate_btn = gr.Button(
                        "✨ Generate Captions",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output section
                    gr.Markdown("### 📝 Generated Captions")
                    
                    with gr.Group():
                        gr.Markdown("**🤖 BLIP Caption**")
                        blip_output = gr.Textbox(
                            label="",
                            placeholder="BLIP caption will appear here...",
                            lines=3,
                            show_copy_button=True
                        )
                    
                    with gr.Group():
                        gr.Markdown("**🤖 GIT Caption**")
                        git_output = gr.Textbox(
                            label="",
                            placeholder="GIT caption will appear here...",
                            lines=3,
                            show_copy_button=True
                        )
            
            # Statistics section
            with gr.Row():
                stats_display = gr.Markdown(
                    value=self.analytics.get_display_stats(),
                    elem_id="stats-display"
                )
            
            # Examples section (if examples exist)
            examples_dir = ui_config.EXAMPLES_DIR
            if examples_dir.exists() and list(examples_dir.glob("*.jpg")):
                gr.Markdown("### 💡 Try These Examples")
                gr.Examples(
                    examples=[str(p) for p in examples_dir.glob("*.jpg")[:3]],
                    inputs=image_input,
                    label=""
                )
            
            # Footer
            gr.Markdown(
                """
                ---
                <div style='text-align: center; color: #666; font-size: 0.9em;'>
                    <p>🚀 Powered by BLIP, GIT, and Groq API | Built with ❤️ using Gradio</p>
                    <p>⚡ Free and Open Source | 📊 All processing done securely</p>
                </div>
                """,
                elem_id="footer"
            )
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_captions,
                inputs=[image_input, style_dropdown],
                outputs=[blip_output, git_output, stats_display],
                api_name="generate"
            )
        
        return interface
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface"""
        return """
        #stats-display {
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            text-align: center;
            font-weight: 500;
            margin: 20px 0;
        }
        
        #footer {
            margin-top: 30px;
        }
        
        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            border: none !important;
            font-weight: 600 !important;
        }
        
        .gr-button-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        """
    
    def launch(
        self,
        share: bool = False,
        server_name: str = "0.0.0.0",
        server_port: int = 7860
    ):
        """
        Launch the Gradio interface
        
        Args:
            share: Create public URL
            server_name: Server host
            server_port: Server port
        """
        interface = self.create_interface()
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            show_api=ui_config.SHOW_API,
            show_error=ui_config.SHOW_ERROR
        )


def main():
    """Main entry point"""
    try:
        app = CaptionGeneratorApp()
        app.launch(
            share=False,  # Set to True to create public URL
            server_name="0.0.0.0",
            server_port=7860
        )
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()