#!/usr/bin/env python3
# SDXL WebUI Flask - Google Colab Setup Script

import os
import sys
import subprocess
import warnings
import time
from tqdm import tqdm

# Ignore warnings
warnings.filterwarnings("ignore")

def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_command(cmd, desc=None):
    """Run a shell command with optional description"""
    if desc:
        print(f"\n>> {desc}...")
    
    try:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        
        # Create progress bar for longer-running commands
        pbar = tqdm(total=100, desc=desc if desc else cmd)
        pbar.update(10)  # Start at 10%
        
        # Stream output
        output = []
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                output.append(line.rstrip())
                # Update progress bar occasionally
                pbar.update(1)
        
        pbar.update(100)  # Complete the progress bar
        pbar.close()
        
        # Check if command was successful
        if process.returncode != 0:
            print(f"Command failed with code {process.returncode}")
            print("\n".join(output[-10:]))  # Print last 10 lines of output
            return False
        
        return True
    except Exception as e:
        print(f"Error running command: {e}")
        return False

def setup_environment():
    """Install required packages and set up the environment"""
    print_header("Setting up SDXL WebUI Flask in Google Colab")
    
    # Check if running in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
    except ImportError:
        print("❌ This script is designed to be run in Google Colab")
        return False
    
    # Install requirements
    print("\n>> Installing required packages...")
    packages = [
        "flask",
        "diffusers",
        "transformers",
        "accelerate",
        "pyngrok",
        "basicsr<1.4.2",
        "realesrgan",
        "opencv-python",
        "controlnet-aux",
        "huggingface_hub",
        "pillow",
        "scipy",
        "ftfy",
        "omegaconf"
    ]
    
    for package in tqdm(packages, desc="Installing packages"):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        except:
            print(f"❌ Failed to install {package}")
    
    # Install PyTorch with CUDA
    run_command("pip install -q torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118", 
                "Installing PyTorch with CUDA")
    
    # Create directory structure
    print("\n>> Creating directory structure...")
    directories = [
        "static",
        "static/css",
        "static/js",
        "templates",
        "models",
        "loras",
        "images",
        "videos",
        "uploads",
        "controlnet",
        "upscaler",
        "CSB",
        "CSB/images",
        "CSB/videos"
    ]
    
    for directory in tqdm(directories, desc="Creating directories"):
        os.makedirs(directory, exist_ok=True)
    
    # Mount Google Drive
    print("\n>> Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Create SDXL directory in Drive if it doesn't exist
        sdxl_dir = '/content/drive/MyDrive/SDXL_WebUI'
        os.makedirs(sdxl_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['images', 'videos', 'loras', 'vae', 'controlnet']:
            os.makedirs(os.path.join(sdxl_dir, subdir), exist_ok=True)
        
        print(f"✓ Google Drive mounted and directories created at {sdxl_dir}")
    except Exception as e:
        print(f"❌ Failed to mount Google Drive: {e}")
    
    # Create .env file for environment variables
    print("\n>> Creating environment configuration...")
    env_content = """# SDXL WebUI Flask Environment Variables
# Fill in your tokens below

# Hugging Face token (for downloading models)
HF_TOKEN=

# CivitAI API key (for searching and downloading LoRAs)
CIVITAI_API_KEY=

# Ngrok token (for exposing the app to the internet)
NGROK_TOKEN=

# Telegram integration
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHANNEL_ID=
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    # Create a README
    print("\n>> Creating README...")
    readme_content = """# SDXL WebUI Flask - Google Colab

This is a Colab-optimized version of SDXL WebUI Flask. To use:

1. Run this setup script
2. Edit the `.env` file to add your API tokens
3. Run the app
4. Access the UI through the provided links

The web interface will be available at:
- Local URL: http://127.0.0.1:7860
- Public URL will be displayed when the app starts (using ngrok)

"""
    
    with open("README_COLAB.md", "w") as f:
        f.write(readme_content)
    
    print("\n✓ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit the .env file to add your API tokens")
    print("2. Run the application with: python app.py")
    
    return True

if __name__ == "__main__":
    setup_environment() 