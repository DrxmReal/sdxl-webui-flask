from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect
import os
import re
import gc
import random
import torch
import json
import time
from datetime import datetime
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np
import requests
import base64
import io
from werkzeug.utils import secure_filename
import threading
import uuid
from functools import wraps

# Import the SDXL components from main.py
from main import (
    load_pipeline, switch_model, load_lora_weights, load_vae, process_controlnet_image,
    load_pipeline_with_controlnet, load_controlnet, upscale_image, add_watermark,
    tokenizer, get_token_count, set_scheduler, load_upscaler, get_ip_address,
    save_to_csb, send_to_telegram, set_huggingface_token, generate_video
)

# Global variables (import or redefine from main.py)
selected_model = "PRIMAGEN/Nova-Furry-XL-V7.B"
selected_vae = "hakurei/waifu-diffusion-vae"
selected_lora = "goofyai/SDXL_Anime_Style"
lora_weight = 0.8
current_model_id = "PRIMAGEN/Nova-Furry-XL-V4.0"
pipe = None
watermark_enabled = False
watermark_text = "SDXL WebUI"
watermark_opacity = 0.3
huggingface_token = os.environ.get("HF_TOKEN", "")  # Get token from environment variable
loaded_loras = {}
vae_model = None
token_limit = 150
current_scheduler = "Euler"
upscaler_model = None
current_controlnet = None

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("videos", exist_ok=True)
os.makedirs("loras", exist_ok=True)
os.makedirs("CSB", exist_ok=True)
os.makedirs("CSB/images", exist_ok=True)
os.makedirs("CSB/videos", exist_ok=True)

# Job management system for long-running tasks
jobs = {}

def initialize_model():
    """Initialize the model at startup"""
    global pipe, selected_model
    if pipe is None:
        print("Initializing model...")
        pipe = load_pipeline(False, selected_model)
        if selected_vae and selected_vae != "None":
            load_vae(selected_vae)
        return True
    return False

# Background task handler
def run_in_background(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "result": None,
            "error": None,
            "start_time": time.time()
        }
        
        def task():
            try:
                result = func(*args, **kwargs)
                jobs[job_id]["result"] = result
                jobs[job_id]["status"] = "completed"
            except Exception as e:
                jobs[job_id]["error"] = str(e)
                jobs[job_id]["status"] = "failed"
            jobs[job_id]["end_time"] = time.time()
        
        thread = threading.Thread(target=task)
        thread.start()
        return job_id
    
    return wrapper

@run_in_background
def generate_images_task(prompt, negative_prompt, width, height, steps, guidance,
                      strength, seed, init_image_path, use_img2img, num_images, webhook_url,
                      do_upscale=False, apply_watermark=False):
    global pipe, selected_model, selected_lora, lora_weight
    
    logs = []
    
    # Initialize or switch model if needed
    model_status = switch_model(selected_model)
    logs.append(model_status)
    
    # Reload pipeline if needed
    if pipe.__class__.__name__.startswith("StableDiffusionXLImg2Img") != use_img2img:
        pipe = load_pipeline(use_img2img, selected_model)
    
    # Apply LoRA if selected
    lora_message = "No LoRA selected"
    if selected_lora and selected_lora != "None":
        pipe, lora_message = load_lora_weights(pipe, selected_lora, lora_weight)
        logs.append(lora_message)
    
    if seed == 0:
        seed = random.randint(1, 1 << 30)
    
    generator = torch.Generator(device="cuda").manual_seed(seed)
    output_dir = os.path.join("images", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle init image for img2img
    init_image = None
    if use_img2img and init_image_path:
        init_image = Image.open(init_image_path).convert("RGB")
    
    previews = []
    output_paths = []
    for i in range(num_images):
        start_time = time.time()
        
        if use_img2img and init_image:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance,
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]
        
        elapsed = time.time() - start_time
        percent = int((i + 1) / num_images * 100)
        
        timestamp = datetime.now().strftime("%H%M%S_%f")
        clean_prompt_text = re.sub(r'[^a-zA-Z0-9]+', '_', prompt.lower())[:50]
        filename = f"{clean_prompt_text}_seed{seed}_{timestamp}.png"
        file_path = os.path.join(output_dir, filename)
        
        bordered_image = ImageOps.expand(image, border=12, fill='black')
        bordered_image.save(file_path)
        
        # Apply upscaler and watermark if requested
        if do_upscale or apply_watermark:
            orig_watermark_enabled = watermark_enabled
            watermark_enabled = apply_watermark
            
            processed_image, process_msg = process_with_upscaler_and_watermark(bordered_image)
            logs.append(process_msg)
            
            processed_filename = f"{clean_prompt_text}_seed{seed}_{timestamp}_processed.png"
            processed_file_path = os.path.join(output_dir, processed_filename)
            processed_image.save(processed_file_path)
            
            # Save to CSB
            csb_result = save_to_csb(processed_file_path)
            logs.append(csb_result)
            
            # Create preview
            output_paths.append(processed_file_path)
            
            # Restore watermark enabled
            watermark_enabled = orig_watermark_enabled
        else:
            # Save to CSB
            csb_result = save_to_csb(file_path)
            logs.append(csb_result)
            output_paths.append(file_path)
        
        # Metadata for webhook/telegram
        metadata = (
            f"🖼️ *Image Generated*\n"
            f"> Prompt: `{prompt}`\n"
            f"> Seed: `{seed}`\n"
            f"> Size: {width}x{height} | Steps: {steps} | Guidance: {guidance}\n"
            f"> Strength: {strength if use_img2img else 'N/A'}\n"
            f"> LoRA: {lora_message if selected_lora != 'None' else 'None'}"
        )
        
        # Send to webhook or Telegram if configured
        if webhook_url:
            try:
                upload_to_webhook(file_path, webhook_url, metadata)
            except Exception as e:
                logs.append(f"⚠️ Webhook error: {str(e)}")
        
        try:
            send_to_telegram(file_path, metadata)
        except Exception as e:
            logs.append(f"⚠️ Telegram error: {str(e)}")
        
        logs.append(f"✅ [{percent}%] Saved: {file_path} | Time: {elapsed:.2f}s")
    
    # Reset LoRA fusion if one was applied
    if selected_lora and selected_lora != "None":
        pipe.unfuse_lora()
    
    return {
        "output_paths": output_paths,
        "logs": logs,
        "seed": seed
    }

def process_with_upscaler_and_watermark(image):
    """Process image with upscaler and watermark"""
    result_message = ""
    
    # Upscale image if upscaler is loaded
    if upscaler_model is not None:
        image, upscale_msg = upscale_image(image)
        result_message += upscale_msg + "\n"
    
    # Add watermark if enabled
    if watermark_enabled:
        image = add_watermark(image)
        result_message += f"✅ Added watermark. Text: '{watermark_text}'\n"
    
    return image, result_message

# Upload webhook (from main.py)
def upload_to_webhook(image_path, webhook_url, metadata_text):
    if not webhook_url.strip():
        return
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            data = {'content': metadata_text}
            r = requests.post(webhook_url, data=data, files=files)
            if r.status_code not in [200, 204]:
                print(f"❌ Upload failed: {r.status_code}")
    except Exception as e:
        print(f"🚨 Upload error: {e}")

def image_to_base64(img_path):
    """Convert an image to base64 for embedding in HTML"""
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/txt2img')
def txt2img():
    """Text to image page"""
    return render_template('txt2img.html')

@app.route('/img2img')
def img2img():
    """Image to image page"""
    return render_template('img2img.html')

@app.route('/controlnet')
def controlnet():
    """ControlNet page"""
    return render_template('controlnet.html')

@app.route('/gallery')
def gallery():
    """Gallery page"""
    # Scan images directory
    image_paths = []
    for date_dir in os.listdir("images"):
        date_path = os.path.join("images", date_dir)
        if os.path.isdir(date_path):
            for img in os.listdir(date_path):
                if img.endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(date_path, img))
    
    image_paths.sort(key=os.path.getmtime, reverse=True)
    
    # Convert to usable format for template
    images = []
    for path in image_paths[:20]:  # Limit to 20 recent images
        rel_path = os.path.relpath(path, start='.')
        filename = os.path.basename(path)
        size = os.path.getsize(path) // 1024  # KB
        mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%Y-%m-%d %H:%M:%S')
        
        # Extract info from filename
        seed = "Unknown"
        seed_match = re.search(r'_seed(\d+)_', filename)
        if seed_match:
            seed = seed_match.group(1)
        
        images.append({
            'path': rel_path,
            'filename': filename,
            'size': size,
            'mtime': mtime,
            'seed': seed
        })
    
    return render_template('gallery.html', images=images)

@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')

@app.route('/video')
def video():
    """Image to video page"""
    return render_template('video.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API endpoint for image generation"""
    try:
        data = request.json
        
        # Extract parameters
        prompt = data.get('prompt', '')
        negative_prompt = data.get('negative_prompt', '')
        width = int(data.get('width', 768))
        height = int(data.get('height', 1024))
        steps = int(data.get('steps', 30))
        guidance = float(data.get('guidance', 7.5))
        seed = int(data.get('seed', 0))
        num_images = int(data.get('num_images', 1))
        strength = float(data.get('strength', 0.6))
        webhook_url = data.get('webhook_url', '')
        do_upscale = bool(data.get('do_upscale', False))
        apply_watermark = bool(data.get('apply_watermark', False))
        
        # Init image handling for img2img
        use_img2img = False
        init_image_path = None
        
        if 'init_image' in request.files:
            init_file = request.files['init_image']
            if init_file.filename:
                use_img2img = True
                init_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(init_file.filename))
                init_file.save(init_image_path)
        
        # Start background task
        job_id = generate_images_task(
            prompt, negative_prompt, width, height, steps, guidance,
            strength, seed, init_image_path, use_img2img, num_images,
            webhook_url, do_upscale, apply_watermark
        )
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Generation started'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/job/<job_id>', methods=['GET'])
def job_status(job_id):
    """Get status of a job"""
    if job_id not in jobs:
        return jsonify({'success': False, 'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    
    response = {
        'status': job['status'],
        'progress': job['progress']
    }
    
    if job['status'] == 'completed':
        result = job['result']
        
        # For image generation, include base64 of images
        if 'output_paths' in result:
            images = []
            for path in result['output_paths']:
                images.append({
                    'path': path,
                    'base64': image_to_base64(path),
                    'filename': os.path.basename(path)
                })
            response['images'] = images
        
        response['logs'] = result.get('logs', [])
        response['seed'] = result.get('seed', 0)
    
    elif job['status'] == 'failed':
        response['error'] = job['error']
    
    # Add timing info
    if 'start_time' in job:
        response['duration'] = time.time() - job['start_time']
    
    return jsonify(response)

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get current settings"""
    return jsonify({
        'selected_model': selected_model,
        'selected_vae': selected_vae,
        'selected_lora': selected_lora,
        'lora_weight': lora_weight,
        'watermark_enabled': watermark_enabled,
        'watermark_text': watermark_text,
        'watermark_opacity': watermark_opacity,
        'huggingface_token': '●●●●●●●●●●●●' if huggingface_token else '',
        'current_scheduler': current_scheduler
    })

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update settings"""
    global selected_model, selected_vae, selected_lora, lora_weight
    global watermark_enabled, watermark_text, watermark_opacity
    global huggingface_token, current_scheduler
    
    data = request.json
    
    # Update settings selectively
    if 'selected_model' in data:
        selected_model = data['selected_model']
    
    if 'selected_vae' in data:
        selected_vae = data['selected_vae']
        load_vae(selected_vae)
    
    if 'selected_lora' in data:
        selected_lora = data['selected_lora']
    
    if 'lora_weight' in data:
        lora_weight = float(data['lora_weight'])
    
    if 'watermark_enabled' in data:
        watermark_enabled = bool(data['watermark_enabled'])
    
    if 'watermark_text' in data:
        watermark_text = data['watermark_text']
    
    if 'watermark_opacity' in data:
        watermark_opacity = float(data['watermark_opacity'])
    
    if 'huggingface_token' in data and data['huggingface_token']:
        huggingface_token = data['huggingface_token']
        set_huggingface_token(huggingface_token)
    
    if 'current_scheduler' in data:
        current_scheduler = data['current_scheduler']
        set_scheduler(current_scheduler)
    
    return jsonify({'success': True, 'message': 'Settings updated'})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    # These would normally be fetched from the main.py constants
    models = [
        "PRIMAGEN/Nova-Furry-XL-V7.B",
        "stabilityai/stable-diffusion-xl-base-1.0",
        "stablediffusionapi/cyberrealistic-v4.0",
        "SG161222/RealVisXL_V3.0",
        "Lykon/dreamshaper-xl-1-0",
        "RunDiffusion/Juggernaut-XL-v7",
        "zenless-lab/sdxl-anything-xl"
    ]
    
    return jsonify({'models': models})

@app.route('/api/vaes', methods=['GET'])
def get_vaes():
    """Get available VAEs"""
    vaes = [
        "None",
        "stabilityai/sd-vae-ft-mse",
        "stabilityai/sdxl-vae",
        "Linaqruf/anime-vae",
        "stablediffusionapi/anything-v5-vae",
        "hakurei/waifu-diffusion-vae"
    ]
    
    return jsonify({'vaes': vaes})

@app.route('/api/loras', methods=['GET'])
def get_loras():
    """Get available LoRAs"""
    global loaded_loras
    
    # Add any loaded LoRAs
    loras = [{"id": "None", "name": "None"}]
    for lora_id, lora_info in loaded_loras.items():
        loras.append({
            "id": lora_id,
            "name": lora_info["name"],
            "path": lora_info["path"]
        })
    
    return jsonify({'loras': loras})

@app.route('/api/schedulers', methods=['GET'])
def get_schedulers():
    """Get available schedulers"""
    schedulers = [
        "Euler",
        "Euler a",
        "DPM++ 2M",
        "DDIM",
        "Heun",
        "KDPM2",
        "LMS",
        "PNDM",
        "UniPC"
    ]
    
    return jsonify({'schedulers': schedulers})

@app.route('/api/tokens', methods=['POST'])
def count_tokens():
    """Count tokens in a prompt"""
    data = request.json
    prompt = data.get('prompt', '')
    
    token_count = get_token_count(prompt)
    return jsonify({'token_count': token_count})

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve an image from the images directory"""
    return send_from_directory('images', filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve a static file"""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Initialize model in a separate thread to not block app startup
    threading.Thread(target=initialize_model).start()
    app.run(host='0.0.0.0', port=5000, debug=True) 