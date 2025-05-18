#!/usr/bin/env python3
# SDXL WebUI Flask - Download & Setup Script
# This script helps set up the necessary environment for the SDXL WebUI application

import os
import sys
import shutil
import subprocess
import argparse
import json
import time
from zipfile import ZipFile
from pathlib import Path
import urllib.request
import platform
from tqdm import tqdm

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Define colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Configuration
DEFAULT_CONFIG = {
    "huggingface_token": "",
    "civitai_api_key": "",
    "ngrok_token": "",
    "telegram_bot_token": "",
    "telegram_channel_id": "",
    "default_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "default_vae": "stabilityai/sdxl-vae",
    "default_lora": "None",
    "download_upscalers": True,
    "download_controlnet": True,
    "create_environment": True
}

# Progress bar for downloads
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    print(f"{Colors.BLUE}Downloading {url}{Colors.ENDC}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def create_directory_structure():
    """Create the necessary directory structure for the application"""
    print(f"{Colors.CYAN}Creating directory structure...{Colors.ENDC}")
    
    # Main directories
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
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"{Colors.GREEN}✓ Created directory: {directory}{Colors.ENDC}")

def setup_static_files():
    """Copy or create static files if they don't exist"""
    print(f"{Colors.CYAN}Setting up static files...{Colors.ENDC}")
    
    # CSS file creation
    css_content = """
:root {
    --primary-color: #00b4d8;
    --secondary-color: #90e0ef;
    --bg-color: #0d1117;
    --content-bg: #161b22;
    --text-color: #e6edf3;
    --border-color: #30363d;
    --hover-color: #21262d;
    --sidebar-width: 240px;
    --header-height: 60px;
    --success-color: #3fb950;
    --warning-color: #f7b955;
    --error-color: #f85149;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: var(--bg-color);
    color: var(--text-color);
    line-height: 1.6;
}

/* App Layout */
.app-container {
    display: flex;
    min-height: 100vh;
}

/* Sidebar */
.sidebar {
    width: var(--sidebar-width);
    background-color: var(--bg-color);
    border-right: 1px solid var(--border-color);
    position: fixed;
    height: 100vh;
    z-index: 100;
    display: flex;
    flex-direction: column;
}

.logo-container {
    padding: 20px;
    border-bottom: 1px solid var(--border-color);
}

.logo-container h1 {
    color: var(--primary-color);
    font-size: 1.5rem;
    text-align: center;
}

.nav-links {
    list-style: none;
    padding: 20px 0;
    flex: 1;
}

.nav-links li {
    padding: 0;
    margin-bottom: 5px;
}

.nav-links a {
    display: block;
    padding: 12px 20px;
    color: var(--text-color);
    text-decoration: none;
    transition: all 0.3s ease;
    border-radius: 6px;
    margin: 0 10px;
}

.nav-links a:hover, .nav-links a.active {
    background-color: var(--hover-color);
    color: var(--primary-color);
}

.nav-links a i {
    margin-right: 10px;
    width: 20px;
    text-align: center;
}

/* Main Content */
.content {
    flex: 1;
    margin-left: var(--sidebar-width);
    display: flex;
    flex-direction: column;
}

.content-header {
    height: var(--header-height);
    padding: 0 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border-color);
    background-color: var(--content-bg);
}

.content-body {
    padding: 20px;
    flex: 1;
    background-color: var(--content-bg);
    min-height: calc(100vh - var(--header-height));
}

/* Cards */
.card {
    background-color: var(--bg-color);
    border-radius: 8px;
    border: 1px solid var(--border-color);
    padding: 20px;
    margin-bottom: 20px;
}

/* Forms */
.form-group {
    margin-bottom: 15px;
}

label {
    display: block;
    margin-bottom: 5px;
    font-weight: 500;
}

input, textarea, select {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--border-color);
    background-color: var(--hover-color);
    color: var(--text-color);
    border-radius: 4px;
    font-size: 0.9rem;
}

/* Buttons */
.btn {
    display: inline-block;
    padding: 10px 20px;
    background-color: var(--hover-color);
    color: var(--text-color);
    border: 1px solid var(--border-color);
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
    transition: all 0.3s ease;
}

.btn:hover {
    background-color: var(--primary-color);
    color: white;
}

.btn-primary {
    background-color: var(--primary-color);
    color: white;
    border: none;
}
"""
    
    js_content = """
// Main JavaScript file for SDXL WebUI

document.addEventListener('DOMContentLoaded', function() {
    // Load settings and update UI
    fetchSettings();
    
    // Setup tabs if present
    setupTabs();
});

// API Handlers
async function fetchSettings() {
    try {
        const response = await fetch('/api/settings');
        const data = await response.json();
        
        // Update model info in header
        const modelInfoElement = document.getElementById('current-model');
        if (modelInfoElement) {
            modelInfoElement.textContent = `Model: ${data.selected_model}`;
        }
    } catch (error) {
        console.error('Error fetching settings:', error);
    }
}

// UI Helpers
function setupTabs() {
    const tabGroups = document.querySelectorAll('.tabs');
    
    tabGroups.forEach(tabGroup => {
        const tabs = tabGroup.querySelectorAll('.tab');
        const tabContents = document.querySelectorAll('.tab-content');
        
        tabs.forEach(tab => {
            tab.addEventListener('click', function() {
                // Remove active class from all tabs
                tabs.forEach(t => t.classList.remove('active'));
                
                // Add active class to current tab
                this.classList.add('active');
                
                // Hide all tab contents
                tabContents.forEach(content => content.classList.remove('active'));
                
                // Show the associated tab content
                const targetId = this.dataset.target;
                if (targetId) {
                    const targetContent = document.getElementById(targetId);
                    if (targetContent) {
                        targetContent.classList.add('active');
                    }
                }
            });
        });
    });
}

function showLoading(message = 'Processing...') {
    const loadingOverlay = document.getElementById('loading-overlay');
    const loadingMessage = document.getElementById('loading-message');
    
    if (loadingOverlay && loadingMessage) {
        loadingMessage.textContent = message;
        loadingOverlay.classList.remove('hidden');
    }
}

function hideLoading() {
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (loadingOverlay) {
        loadingOverlay.classList.add('hidden');
    }
}

function showNotification(message, type = 'info') {
    alert(message);
}
"""
    
    # Write CSS file
    with open("static/css/style.css", "w") as f:
        f.write(css_content)
    print(f"{Colors.GREEN}✓ Created static/css/style.css{Colors.ENDC}")
    
    # Write JS file
    with open("static/js/main.js", "w") as f:
        f.write(js_content)
    print(f"{Colors.GREEN}✓ Created static/js/main.js{Colors.ENDC}")
    
    # Create base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}SDXL WebUI{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('serve_static', filename='css/style.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    {% block additional_css %}{% endblock %}
</head>
<body>
    <div class="app-container">
        <nav class="sidebar">
            <div class="logo-container">
                <h1>SDXL WebUI</h1>
            </div>
            <ul class="nav-links">
                <li><a href="{{ url_for('index') }}" class="{% if request.path == url_for('index') %}active{% endif %}">
                    <i class="fas fa-home"></i> Dashboard</a>
                </li>
                <li><a href="{{ url_for('txt2img') }}" class="{% if request.path == url_for('txt2img') %}active{% endif %}">
                    <i class="fas fa-pencil-alt"></i> Text to Image</a>
                </li>
                <li><a href="{{ url_for('img2img') }}" class="{% if request.path == url_for('img2img') %}active{% endif %}">
                    <i class="fas fa-image"></i> Image to Image</a>
                </li>
                <li><a href="{{ url_for('gallery') }}" class="{% if request.path == url_for('gallery') %}active{% endif %}">
                    <i class="fas fa-images"></i> Gallery</a>
                </li>
                <li><a href="{{ url_for('settings') }}" class="{% if request.path == url_for('settings') %}active{% endif %}">
                    <i class="fas fa-cog"></i> Settings</a>
                </li>
            </ul>
            <div class="sidebar-footer">
                <p>v1.0.0 | <a href="https://github.com/DrxmReal/sdxl-webui-flask" target="_blank" rel="noopener">GitHub</a></p>
            </div>
        </nav>

        <main class="content">
            <header class="content-header">
                <h2>{% block header_title %}SDXL WebUI{% endblock %}</h2>
                <div class="user-actions">
                    <div class="model-info">
                        <span id="current-model">Loading model...</span>
                    </div>
                </div>
            </header>

            <div class="content-body">
                {% block content %}{% endblock %}
            </div>
        </main>
    </div>

    <!-- Loading spinner -->
    <div id="loading-overlay" class="loading-overlay hidden">
        <div class="spinner"></div>
        <div id="loading-message">Processing...</div>
    </div>

    <script src="{{ url_for('serve_static', filename='js/main.js') }}"></script>
    {% block additional_js %}{% endblock %}
</body>
</html>"""
    
    # Create index template
    index_template = """{% extends "base.html" %}

{% block title %}SDXL WebUI - Dashboard{% endblock %}

{% block header_title %}Dashboard{% endblock %}

{% block content %}
<div class="dashboard-container">
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Welcome to SDXL WebUI</h3>
        </div>
        <div class="card-body">
            <p>This web interface allows you to use Stable Diffusion XL for image generation.</p>
            <p>Select an option from the sidebar to get started.</p>
        </div>
    </div>
</div>
{% endblock %}"""
    
    # Write templates
    with open("templates/base.html", "w") as f:
        f.write(base_template)
    print(f"{Colors.GREEN}✓ Created templates/base.html{Colors.ENDC}")
    
    with open("templates/index.html", "w") as f:
        f.write(index_template)
    print(f"{Colors.GREEN}✓ Created templates/index.html{Colors.ENDC}")

def install_requirements():
    """Install required packages"""
    print(f"{Colors.CYAN}Installing required packages...{Colors.ENDC}")
    
    requirements = [
        "flask>=2.0.0",
        "torch>=2.0.0",
        "numpy>=1.20.0",
        "pillow>=9.0.0",
        "diffusers>=0.18.0",
        "transformers>=4.30.0",
        "accelerate>=0.20.0",
        "requests>=2.25.0",
        "pyngrok>=5.0.0",
        "tqdm>=4.64.0",
        "huggingface-hub>=0.14.0"
    ]
    
    # Write requirements.txt
    with open("requirements.txt", "w") as f:
        f.write("\n".join(requirements))
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print(f"{Colors.GREEN}✓ Successfully installed required packages{Colors.ENDC}")
    except subprocess.CalledProcessError as e:
        print(f"{Colors.FAIL}Failed to install required packages: {e}{Colors.ENDC}")
        print(f"{Colors.WARNING}Please install them manually using: pip install -r requirements.txt{Colors.ENDC}")

def create_config_file(config):
    """Create a config file with user settings"""
    print(f"{Colors.CYAN}Creating configuration file...{Colors.ENDC}")
    
    # Write config to file
    with open("config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"{Colors.GREEN}✓ Created config.json{Colors.ENDC}")
    
    # Also create a .env file for environment variables
    with open(".env", "w") as f:
        f.write(f"HF_TOKEN={config['huggingface_token']}\n")
        f.write(f"CIVITAI_API_KEY={config['civitai_api_key']}\n")
        f.write(f"NGROK_TOKEN={config['ngrok_token']}\n")
        f.write(f"TELEGRAM_BOT_TOKEN={config['telegram_bot_token']}\n")
        f.write(f"TELEGRAM_CHANNEL_ID={config['telegram_channel_id']}\n")
    
    print(f"{Colors.GREEN}✓ Created .env file{Colors.ENDC}")

def create_startup_script():
    """Create a startup script based on platform"""
    print(f"{Colors.CYAN}Creating startup script...{Colors.ENDC}")
    
    if platform.system() == "Windows":
        with open("start.bat", "w") as f:
            f.write('@echo off\n')
            f.write('echo Loading SDXL WebUI...\n')
            f.write('python app.py\n')
            f.write('pause\n')
        print(f"{Colors.GREEN}✓ Created start.bat{Colors.ENDC}")
    else:
        with open("start.sh", "w") as f:
            f.write('#!/bin/bash\n')
            f.write('echo "Loading SDXL WebUI..."\n')
            f.write('python3 app.py\n')
        os.chmod("start.sh", 0o755)  # Make executable
        print(f"{Colors.GREEN}✓ Created start.sh{Colors.ENDC}")

def create_colab_notebook():
    """Create a Google Colab notebook for easy setup"""
    print(f"{Colors.CYAN}Creating Google Colab notebook...{Colors.ENDC}")
    
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# SDXL WebUI Flask - Google Colab\n", 
                          "This notebook helps you run SDXL WebUI Flask in Google Colab."]
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Clone the repository\n",
                    "!git clone https://github.com/DrxmReal/sdxl-webui-flask.git\n",
                    "%cd sdxl-webui-flask"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Install dependencies\n",
                    "!pip install -r requirements.txt"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Set environment variables\n",
                    "import os\n",
                    "os.environ[\"HF_TOKEN\"] = \"\"  # Your Hugging Face token\n",
                    "os.environ[\"CIVITAI_API_KEY\"] = \"\"  # Your CivitAI API key\n",
                    "os.environ[\"NGROK_TOKEN\"] = \"\"  # Your Ngrok token\n",
                    "os.environ[\"TELEGRAM_BOT_TOKEN\"] = \"\"  # Your Telegram Bot token\n",
                    "os.environ[\"TELEGRAM_CHANNEL_ID\"] = \"\"  # Your Telegram channel ID"
                ],
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": [
                    "# Run the application\n",
                    "!python app.py"
                ],
                "execution_count": None,
                "outputs": []
            }
        ],
        "metadata": {
            "colab": {
                "gpuType": "T4",
                "provenance": []
            },
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3"
            },
            "language_info": {
                "name": "python"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    with open("sdxl_webui_colab.ipynb", "w") as f:
        json.dump(notebook_content, f, indent=2)
    
    print(f"{Colors.GREEN}✓ Created sdxl_webui_colab.ipynb{Colors.ENDC}")

def create_environment_file():
    """Create a virtual environment setup file"""
    print(f"{Colors.CYAN}Creating virtual environment setup script...{Colors.ENDC}")
    
    if platform.system() == "Windows":
        with open("setup_venv.bat", "w") as f:
            f.write('@echo off\n')
            f.write('echo Creating virtual environment...\n')
            f.write('python -m venv venv\n')
            f.write('echo Activating virtual environment...\n')
            f.write('call venv\\Scripts\\activate\n')
            f.write('echo Installing requirements...\n')
            f.write('pip install -r requirements.txt\n')
            f.write('echo Setup complete!\n')
            f.write('pause\n')
        print(f"{Colors.GREEN}✓ Created setup_venv.bat{Colors.ENDC}")
    else:
        with open("setup_venv.sh", "w") as f:
            f.write('#!/bin/bash\n')
            f.write('echo "Creating virtual environment..."\n')
            f.write('python3 -m venv venv\n')
            f.write('echo "Activating virtual environment..."\n')
            f.write('source venv/bin/activate\n')
            f.write('echo "Installing requirements..."\n')
            f.write('pip install -r requirements.txt\n')
            f.write('echo "Setup complete!"\n')
        os.chmod("setup_venv.sh", 0o755)  # Make executable
        print(f"{Colors.GREEN}✓ Created setup_venv.sh{Colors.ENDC}")

def get_user_config():
    """Prompt the user for configuration settings"""
    print(f"{Colors.HEADER}SDXL WebUI Setup Configuration{Colors.ENDC}")
    print("Please provide the following information (press Enter to use default/empty values):")
    
    config = DEFAULT_CONFIG.copy()
    
    config["huggingface_token"] = input(f"Hugging Face Token [empty]: ").strip() or ""
    config["civitai_api_key"] = input(f"CivitAI API Key [empty]: ").strip() or ""
    config["ngrok_token"] = input(f"Ngrok Token [empty]: ").strip() or ""
    config["telegram_bot_token"] = input(f"Telegram Bot Token [empty]: ").strip() or ""
    config["telegram_channel_id"] = input(f"Telegram Channel ID [empty]: ").strip() or ""
    
    default_model = config["default_model"]
    model_input = input(f"Default SDXL Model [{default_model}]: ").strip()
    config["default_model"] = model_input if model_input else default_model
    
    return config

def main():
    parser = argparse.ArgumentParser(description="SDXL WebUI Flask Setup Script")
    parser.add_argument("--auto", action="store_true", help="Automatically setup with default values")
    args = parser.parse_args()
    
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.HEADER}SDXL WebUI Flask - Setup Script{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    
    # Get user configuration if not in auto mode
    if args.auto:
        config = DEFAULT_CONFIG
    else:
        config = get_user_config()
    
    # Setup base directories and files
    create_directory_structure()
    setup_static_files()
    
    # Create configuration file
    create_config_file(config)
    
    # Create startup scripts
    create_startup_script()
    
    # Install requirements
    install_requirements()
    
    # Create virtual environment setup script
    if config["create_environment"]:
        create_environment_file()
    
    # Create Colab notebook
    create_colab_notebook()
    
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")
    print(f"{Colors.GREEN}Setup completed successfully!{Colors.ENDC}")
    print(f"{Colors.CYAN}To start the application, run:{Colors.ENDC}")
    if platform.system() == "Windows":
        print(f"{Colors.CYAN}  start.bat{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}  ./start.sh{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*50}{Colors.ENDC}")

if __name__ == "__main__":
    main() 