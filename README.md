---
title: AI Image Caption Generator
emoji: 🖼️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.8.0
app_file: app.py
pinned: false
license: mit
---

# 🖼️ AI Image Caption Generator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/ChinmayM06/ai-image-caption-generator)

> Generate AI-powered image captions with multiple style options—completely free, no API costs.

A lightweight, GPU-accelerated image captioning tool using state-of-the-art vision-language models (BLIP & GIT) with style customization powered by Groq's free LLM API.

---

## ✨ Features

- 🎯 **Dual Model Support**: Both BLIP-base (fast) and GIT-large (high quality) run simultaneously
- 🎨 **5 Caption Styles**: None, Creative, Social Media, Professional, Technical
- ⚡ **GPU Accelerated**: Optimized for NVIDIA GPUs (works on CPU too)
- 📊 **Analytics Tracking**: Built-in usage statistics and performance metrics
- 🖼️ **Image Processing**: Automatic validation, resizing, and format conversion
- 🔄 **Fallback Mechanisms**: Graceful degradation when API is unavailable
- 💰 **100% Free**: No OpenAI credits, no hidden costs
- 🔒 **Privacy First**: Local inference option available

---

## 🚀 Live Demo

Try it out without any installation:

**[🎮 Launch Live Demo →](https://huggingface.co/spaces/ChinmayM06/ai-image-caption-generator)**

*Add your Hugging Face Spaces URL above after deployment*

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Vision Models** | BLIP-base, GIT-large (Hugging Face) |
| **Style LLM** | Groq API (free tier) |
| **Framework** | PyTorch 2.1.0 + CUDA 11.8 |
| **Interface** | Gradio 4.8.0 |
| **Deployment** | Hugging Face Spaces (T4 GPU) |

---

## 📦 Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 4GB+ VRAM (recommended) or CPU
- CUDA 11.8 (for GPU acceleration)

### Installation

```bash
# Clone repository
git clone https://github.com/ChinmayM06/ai-image-caption-generator.git
cd ai-image-caption-generator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (optional)
# Create a .env file in the project root with:
# GROQ_API_KEY=your_groq_api_key_here
# Get your free API key at https://console.groq.com
# Note: The app works without API key but styling features will use fallback templates

# Run the application
python app.py
```

Access at `http://localhost:7860`

---

## 🎯 Usage

### Basic Usage

```python
from src.models import get_model_manager, get_style_model
from src.utils import get_image_processor
from PIL import Image

# Initialize components (singleton pattern)
model_manager = get_model_manager()
style_model = get_style_model()
image_processor = get_image_processor()

# Load models (BLIP and GIT)
blip_success, git_success = model_manager.load_all_models()

# Load and preprocess image
image = Image.open("your_image.jpg")
processed_img, metadata = image_processor.preprocess_image(image)

# Generate captions from both models
captions = model_manager.generate_captions(processed_img)
blip_caption = captions["blip"]
git_caption = captions["git"]

# Apply style (optional)
styled_blip = style_model.style_caption(blip_caption, style="Professional")
styled_git = style_model.style_caption(git_caption, style="Creative")
```

### Available Models

Both models run simultaneously to provide comparison:
- **BLIP-base**: Fast inference (~1-2s), good quality, efficient
- **GIT-large**: Slower (~3-4s), superior caption quality, more detailed

### Caption Styles

| Style | Use Case | Example |
|-------|----------|---------|
| **None** | Raw model output | "A dog sitting on grass" |
| **Creative** | Artistic, imaginative | "A joyful golden retriever basking in nature's embrace" |
| **Social Media** | Engaging, hashtag-ready | "Meet this good boy enjoying sunny vibes! 🐕☀️ #DogLife" |
| **Professional** | Business, formal | "Canine subject positioned in outdoor environment" |
| **Technical** | Detailed, analytical | "Golden retriever breed, seated posture, natural lighting, outdoor setting" |

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t caption-generator .

# Run container (with GPU)
docker run --gpus all -p 7860:7860 caption-generator

# Run container (CPU only)
docker run -p 7860:7860 -e DEVICE=cpu caption-generator
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root (optional):

```bash
# Groq API Key (required for advanced styling, fallback available)
GROQ_API_KEY=your_groq_api_key_here

# Hardware Configuration (optional, defaults to 'cuda' if available)
DEVICE=cuda  # or 'cpu'

# Logging Level (optional)
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

---

## 🎓 Why This Project?

Built as a learning project to explore:
- **GenAI Fundamentals**: Vision-language models, prompt engineering
- **Practical ML Skills**: GPU optimization, model deployment, API integration
- **Cost Optimization**: Demonstrating production-quality AI without expensive APIs
- **Software Architecture**: Caching, analytics, error handling, thread safety

Perfect for understanding how modern image captioning works under the hood while keeping infrastructure costs at zero.

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation
- Add new caption styles
- Optimize performance


---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [Salesforce BLIP](https://github.com/salesforce/BLIP) - Image captioning model
- [Microsoft GIT](https://github.com/microsoft/GenerativeImage2Text) - High-quality captions
- [Groq](https://groq.com) - Free LLM inference API
- [Hugging Face](https://huggingface.co) - Model hosting & deployment

---

## 📬 Contact

**Chinmay M** - [@ChinmayM06](https://github.com/ChinmayM06)

Project Link: [https://github.com/ChinmayM06/ai-image-caption-generator](https://github.com/ChinmayM06/ai-image-caption-generator)

---

<div align="center">

**[⭐ Star this repo](https://github.com/ChinmayM06/ai-image-caption-generator)** if you find it helpful!

Made with ❤️ and lots of ☕

</div>