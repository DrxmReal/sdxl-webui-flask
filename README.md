# SDXL WebUI Flask

A modern Flask-based web interface for Stable Diffusion XL with a clean, dark-themed UI.

## Features

- Text to Image generation with Stable Diffusion XL
- Image to Image transformation 
- ControlNet integration (Canny, OpenPose, Depth, MLSD, etc.)
- Image to Video generation
- Gallery for browsing generated images
- Model, VAE, and LoRA configuration
- Upscaling functionality
- Watermarking options

## Setup

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU recommended

### Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/sdxl-webui-flask.git
cd sdxl-webui-flask
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables for API tokens (optional but recommended):
```bash
# Linux/Mac
export HF_TOKEN="your_huggingface_token"
export CIVITAI_API_KEY="your_civitai_api_key"
export NGROK_TOKEN="your_ngrok_token"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
export TELEGRAM_CHANNEL_ID="your_telegram_channel_id"

# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token"
$env:CIVITAI_API_KEY="your_civitai_api_key"
$env:NGROK_TOKEN="your_ngrok_token"
$env:TELEGRAM_BOT_TOKEN="your_telegram_bot_token"
$env:TELEGRAM_CHANNEL_ID="your_telegram_channel_id"
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Google Colab Usage

To use this application in Google Colab:

1. Create a new Colab notebook
2. Add and run the following code:

```python
# Clone the repository
!git clone https://github.com/YOUR_USERNAME/sdxl-webui-flask.git
%cd sdxl-webui-flask

# Install dependencies
!pip install -r requirements.txt

# Set environment variables
import os
os.environ["HF_TOKEN"] = "your_huggingface_token"
os.environ["CIVITAI_API_KEY"] = "your_civitai_api_key"
os.environ["NGROK_TOKEN"] = "your_ngrok_token"
os.environ["TELEGRAM_BOT_TOKEN"] = "your_telegram_bot_token"
os.environ["TELEGRAM_CHANNEL_ID"] = "your_telegram_channel_id"

# Run the application
!python app.py
```

## License

MIT License