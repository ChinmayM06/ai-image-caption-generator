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
        image,
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
                    pil_image = Image.open(image)
                elif isinstance(image, Image.Image):
                    pil_image = image
                elif hasattr(image, 'shape'):
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
        Create Gradio interface with enhanced UI and mobile responsiveness
        
        Returns:
            gr.Blocks: Configured Gradio interface
        """
        with gr.Blocks(
            theme=gr.themes.Soft(
                primary_hue="purple",
                secondary_hue="blue",
                neutral_hue="slate",
            ),
            title=ui_config.TITLE,
            css=self._get_custom_css()
        ) as interface:
            
            # Header with enhanced styling
            with gr.Row(elem_classes="header-container"):
                with gr.Column():
                    gr.HTML("""
                        <div class="app-header">
                            <h1 class="main-title">
                                <span class="emoji-icon">🎨</span>
                                AI Image Caption Generator
                            </h1>
                            <p class="subtitle">
                                Transform your images into compelling captions with AI-powered models
                            </p>
                        </div>
                    """)
            
            # Main content area with mobile-responsive layout
            with gr.Row(elem_classes="main-content"):
                # Left column - Input section
                with gr.Column(scale=1, min_width=300, elem_classes="input-column"):
                    gr.HTML('<div class="section-header"><span class="emoji-icon">📤</span> Upload Image</div>')
                    
                    image_input = gr.Image(
                        label="",
                        type="pil",
                        height=400,
                        elem_classes="image-upload"
                    )
                    
                    gr.HTML('<div class="section-header"><span class="emoji-icon">🎨</span> Choose Style</div>')
                    
                    style_dropdown = gr.Dropdown(
                        choices=self.style_model.get_available_styles(),
                        value="Professional",
                        label="",
                        info="Select how you want your caption to be styled",
                        elem_classes="style-dropdown",
                        allow_custom_value=False
                    )
                    
                    generate_btn = gr.Button(
                        "✨ Generate Captions",
                        variant="primary",
                        size="lg",
                        elem_classes="generate-button"
                    )
                    
                    # Info card
                    gr.HTML("""
                        <div class="info-card">
                            <p><span class="emoji-icon">💡</span> <strong>Tip:</strong> Upload high-quality images for best results</p>
                            <p><span class="emoji-icon">⚡</span> Processing typically takes 3-10 seconds</p>
                        </div>
                    """)
                
                # Right column - Output section
                with gr.Column(scale=1, min_width=300, elem_classes="output-column"):
                    gr.HTML('<div class="section-header"><span class="emoji-icon">📝</span> Generated Captions</div>')
                    
                    with gr.Group(elem_classes="caption-card"):
                        gr.HTML('<div class="model-label"><span class="emoji-icon">🤖</span> BLIP Model</div>')
                        blip_output = gr.Textbox(
                            label="",
                            placeholder="BLIP caption will appear here...",
                            lines=4,
                            show_copy_button=True,
                            elem_classes="caption-output"
                        )
                    
                    with gr.Group(elem_classes="caption-card"):
                        gr.HTML('<div class="model-label"><span class="emoji-icon">🤖</span> GIT Model</div>')
                        git_output = gr.Textbox(
                            label="",
                            placeholder="GIT caption will appear here...",
                            lines=4,
                            show_copy_button=True,
                            elem_classes="caption-output"
                        )
            
            # Statistics section with enhanced styling
            with gr.Row(elem_classes="stats-row"):
                stats_display = gr.HTML(
                    value=f'<div class="stats-display">{self.analytics.get_display_stats()}</div>',
                    elem_id="stats-display"
                )
            
            # Examples section (if examples exist)
            examples_dir = ui_config.EXAMPLES_DIR
            if examples_dir.exists() and list(examples_dir.glob("*.jpg")):
                with gr.Row(elem_classes="examples-section"):
                    with gr.Column():
                        gr.HTML('<div class="section-header"><span class="emoji-icon">💡</span> Try These Examples</div>')
                        gr.Examples(
                            examples=[str(p) for p in examples_dir.glob("*.jpg")[:3]],
                            inputs=image_input,
                            label="",
                            examples_per_page=3
                        )
            
            # Footer with better styling
            gr.HTML("""
                <div class="footer">
                    <div class="footer-content">
                        <p class="footer-main">
                            <span class="emoji-icon">🚀</span> Powered by BLIP, GIT, and Groq API
                        </p>
                        <p class="footer-sub">
                            <span class="emoji-icon">⚡</span> Free & Open Source | 
                            <span class="emoji-icon">🔒</span> Secure Processing | 
                            <span class="emoji-icon">❤️</span> Built with Gradio
                        </p>
                    </div>
                </div>
            """)
            
            # Event handlers
            def generate_with_stats_update(*args):
                blip, git, stats = self.generate_captions(*args)
                stats_html = f'<div class="stats-display">{stats}</div>'
                return blip, git, stats_html
            
            generate_btn.click(
                fn=generate_with_stats_update,
                inputs=[image_input, style_dropdown],
                outputs=[blip_output, git_output, stats_display],
                api_name="generate"
            )
        
        return interface
    
    def _get_custom_css(self) -> str:
        """Get custom CSS for the interface with emoji support and mobile responsiveness"""
        return """
        /* Ensure UTF-8 encoding for emoji support */
        @charset "UTF-8";
        
        /* Global emoji styling */
        .emoji-icon {
            font-family: "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji", sans-serif;
            font-size: 1.2em;
            margin-right: 0.3em;
        }
        
        /* Override Gradio's body background for consistency */
        body {
            background: linear-gradient(to bottom, #0f172a 0%, #1e293b 100%) !important;
        }
        
        /* Header styling - enhanced for dark theme */
        .app-header {
            text-align: center;
            padding: 2.5rem 1.5rem;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #8b5cf6 100%);
            border-radius: 20px;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(139, 92, 246, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            color: white;
            margin: 0;
            text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
            letter-spacing: -0.5px;
        }
        
        .subtitle {
            font-size: 1.15rem;
            color: rgba(255, 255, 255, 0.95);
            margin: 0.75rem 0 0 0;
            font-weight: 400;
        }
        
        /* Section headers - enhanced with gradient */
        .section-header {
            font-size: 1.3rem;
            font-weight: 600;
            color: #f1f5f9;
            margin: 1.5rem 0 1rem 0;
            padding: 0.75rem 0;
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, transparent 100%);
            border-left: 4px solid #8b5cf6;
            padding-left: 1rem;
            border-radius: 4px;
        }
        
        /* Main content area - enhanced depth */
        .main-content {
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .input-column, .output-column {
            padding: 2rem;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(51, 65, 85, 0.95) 100%);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(139, 92, 246, 0.25);
            backdrop-filter: blur(10px);
        }
        
        /* Image upload area - dark theme */
        .image-upload {
            border: 2px dashed rgba(139, 92, 246, 0.4);
            border-radius: 12px;
            transition: all 0.3s ease;
            background: rgba(15, 23, 42, 0.5);
        }
        
        .image-upload:hover {
            border-color: #8b5cf6;
            box-shadow: 0 0 20px rgba(139, 92, 246, 0.3);
            background: rgba(15, 23, 42, 0.7);
        }
        
        /* Style dropdown */
        .style-dropdown {
            margin-bottom: 1rem;
        }
        
        /* Generate button - vibrant gradient */
        .generate-button {
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #8b5cf6 100%) !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 1.1rem !important;
            padding: 0.85rem 2rem !important;
            border-radius: 12px !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5) !important;
            transition: all 0.3s ease !important;
            width: 100% !important;
            color: white !important;
        }
        
        .generate-button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 30px rgba(139, 92, 246, 0.6) !important;
        }
        
        .generate-button:active {
            transform: translateY(0) !important;
        }
        
        /* Info card - dark theme */
        .info-card {
            background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(99, 102, 241, 0.15) 100%);
            padding: 1.25rem;
            border-radius: 12px;
            margin-top: 1.5rem;
            border-left: 4px solid #8b5cf6;
            border: 1px solid rgba(139, 92, 246, 0.3);
        }
        
        .info-card p {
            margin: 0.5rem 0;
            color: #e2e8f0;
            font-size: 0.95rem;
        }
        
        /* Caption cards - enhanced with better visual separation */
        .caption-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(51, 65, 85, 0.9) 100%);
            padding: 1.75rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
            border: 1px solid rgba(139, 92, 246, 0.3);
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .caption-card:hover {
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.25);
            border-color: rgba(139, 92, 246, 0.5);
            transform: translateY(-3px);
            background: linear-gradient(135deg, rgba(30, 41, 59, 1) 0%, rgba(51, 65, 85, 1) 100%);
        }
        
        .model-label {
            font-size: 1.1rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-bottom: 1rem;
            padding: 0.5rem 1rem;
            background: linear-gradient(90deg, rgba(139, 92, 246, 0.25) 0%, rgba(99, 102, 241, 0.15) 100%);
            border-radius: 8px;
            border-left: 3px solid #8b5cf6;
            display: inline-block;
        }
        
        .caption-output {
            font-size: 1rem;
            line-height: 1.7;
            background: rgba(15, 23, 42, 0.7) !important;
            border: 1px solid rgba(139, 92, 246, 0.25) !important;
            color: #e2e8f0 !important;
            border-radius: 10px !important;
            padding: 1rem !important;
        }
        
        .caption-output:focus {
            border-color: rgba(139, 92, 246, 0.5) !important;
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15) !important;
        }
        
        /* Statistics display */
        .stats-display {
            padding: 1.75rem;
            background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 50%, #8b5cf6 100%);
            color: white;
            border-radius: 16px;
            text-align: center;
            font-weight: 500;
            font-size: 1.05rem;
            box-shadow: 0 6px 25px rgba(139, 92, 246, 0.4);
            margin: 2rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Examples section */
        .examples-section {
            margin: 2rem 0;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            padding: 2rem;
            border-radius: 16px;
            border: 1px solid rgba(139, 92, 246, 0.2);
        }
        
        /* Footer - dark theme */
        .footer {
            margin-top: 3rem;
            padding: 2rem 1.5rem;
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(51, 65, 85, 0.8) 100%);
            border-radius: 16px;
            border: 1px solid rgba(139, 92, 246, 0.3);
            border-top: 3px solid #8b5cf6;
        }
        
        .footer-content {
            text-align: center;
        }
        
        .footer-main {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin: 0.5rem 0;
        }
        
        .footer-sub {
            font-size: 0.95rem;
            color: #cbd5e1;
            margin: 0.5rem 0;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .main-title {
                font-size: 1.8rem;
            }
            
            .subtitle {
                font-size: 1rem;
            }
            
            .app-header {
                padding: 1.5rem 1rem;
            }
            
            .main-content {
                flex-direction: column;
                gap: 1rem;
            }
            
            .input-column, .output-column {
                padding: 1.5rem;
                min-width: 100%;
            }
            
            .section-header {
                font-size: 1.1rem;
            }
            
            .generate-button {
                font-size: 1rem !important;
                padding: 0.7rem 1.5rem !important;
            }
            
            .caption-card {
                padding: 1.25rem;
            }
            
            .stats-display {
                font-size: 0.95rem;
                padding: 1.25rem;
            }
            
            .footer {
                padding: 1.5rem 1rem;
            }
            
            .footer-main {
                font-size: 1rem;
            }
            
            .footer-sub {
                font-size: 0.85rem;
            }
        }
        
        @media (max-width: 480px) {
            .main-title {
                font-size: 1.5rem;
            }
            
            .emoji-icon {
                font-size: 1em;
            }
            
            .info-card {
                padding: 1rem;
            }
            
            .info-card p {
                font-size: 0.85rem;
            }
            
            .input-column, .output-column {
                padding: 1rem;
            }
        }
        
        /* Ensure proper spacing on all devices */
        .gr-row {
            margin-bottom: 1rem;
        }
        
        .gr-column {
            padding: 0.5rem;
        }
        
        /* Override Gradio's default styles for dark theme */
        .gr-box {
            border-radius: 12px !important;
            background: transparent !important;
        }
        
        .gr-input, .gr-dropdown {
            border-radius: 8px !important;
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            color: #e2e8f0 !important;
        }
        
        .gr-input:focus, .gr-dropdown:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Gradio Panel styling */
        .gr-panel {
            background: transparent !important;
            border: none !important;
        }
        
        /* Dropdown menu styling */
        .gr-dropdown-menu {
            background: #1e293b !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
        }
        
        .gr-dropdown-menu option {
            background: #1e293b !important;
            color: #e2e8f0 !important;
        }
        
        .gr-dropdown-menu option:hover {
            background: rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Image component styling */
        .gr-image {
            background: rgba(15, 23, 42, 0.5) !important;
            border: 1px solid rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Textbox styling */
        textarea {
            background: rgba(15, 23, 42, 0.6) !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            color: #e2e8f0 !important;
        }
        
        textarea:focus {
            border-color: #8b5cf6 !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
        }
        
        /* Copy button styling */
        .copy-button {
            background: rgba(139, 92, 246, 0.2) !important;
            border: 1px solid rgba(139, 92, 246, 0.3) !important;
            color: #e2e8f0 !important;
        }
        
        .copy-button:hover {
            background: rgba(139, 92, 246, 0.3) !important;
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