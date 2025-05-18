// Main JavaScript file for SDXL WebUI

document.addEventListener('DOMContentLoaded', function() {
    // Load settings and update UI
    fetchSettings();
    
    // Setup event listeners for modals
    setupModalHandlers();
    
    // Initialize tabs if present
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
        
        // Update settings form fields if on settings page
        const settingsForm = document.getElementById('settings-form');
        if (settingsForm) {
            // Model selection
            const modelSelect = document.getElementById('model-select');
            if (modelSelect) {
                modelSelect.value = data.selected_model;
            }
            
            // VAE selection
            const vaeSelect = document.getElementById('vae-select');
            if (vaeSelect) {
                vaeSelect.value = data.selected_vae;
            }
            
            // LoRA selection
            const loraSelect = document.getElementById('lora-select');
            if (loraSelect) {
                loraSelect.value = data.selected_lora;
            }
            
            // LoRA weight
            const loraWeight = document.getElementById('lora-weight');
            if (loraWeight) {
                loraWeight.value = data.lora_weight;
            }
            
            // Watermark settings
            const watermarkEnabled = document.getElementById('watermark-enabled');
            if (watermarkEnabled) {
                watermarkEnabled.checked = data.watermark_enabled;
            }
            
            const watermarkText = document.getElementById('watermark-text');
            if (watermarkText) {
                watermarkText.value = data.watermark_text;
            }
            
            const watermarkOpacity = document.getElementById('watermark-opacity');
            if (watermarkOpacity) {
                watermarkOpacity.value = data.watermark_opacity;
            }
            
            // Scheduler
            const schedulerSelect = document.getElementById('scheduler-select');
            if (schedulerSelect) {
                schedulerSelect.value = data.current_scheduler;
            }
        }
    } catch (error) {
        console.error('Error fetching settings:', error);
    }
}

async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        return data.models;
    } catch (error) {
        console.error('Error fetching models:', error);
        return [];
    }
}

async function fetchVAEs() {
    try {
        const response = await fetch('/api/vaes');
        const data = await response.json();
        return data.vaes;
    } catch (error) {
        console.error('Error fetching VAEs:', error);
        return [];
    }
}

async function fetchLoRAs() {
    try {
        const response = await fetch('/api/loras');
        const data = await response.json();
        return data.loras;
    } catch (error) {
        console.error('Error fetching LoRAs:', error);
        return [];
    }
}

async function fetchSchedulers() {
    try {
        const response = await fetch('/api/schedulers');
        const data = await response.json();
        return data.schedulers;
    } catch (error) {
        console.error('Error fetching schedulers:', error);
        return [];
    }
}

async function saveSettings(settingsData) {
    try {
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settingsData)
        });
        
        const result = await response.json();
        
        if (result.success) {
            showNotification('Settings saved successfully', 'success');
        } else {
            showNotification('Failed to save settings', 'error');
        }
        
        // Refresh settings display
        fetchSettings();
        
    } catch (error) {
        console.error('Error saving settings:', error);
        showNotification('Error saving settings', 'error');
    }
}

// Image Generation
async function generateImage(formData, isImg2Img = false) {
    showLoading('Generating image...');
    
    try {
        const response = await fetch('/api/generate', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Start polling for job status
            pollJobStatus(data.job_id);
        } else {
            hideLoading();
            showNotification('Failed to start generation: ' + data.error, 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Error generating image:', error);
        showNotification('Error generating image', 'error');
    }
}

async function pollJobStatus(jobId) {
    try {
        const response = await fetch(`/api/job/${jobId}`);
        const data = await response.json();
        
        if (data.status === 'running') {
            // Update progress if available
            if (data.progress) {
                updateLoadingProgress(data.progress);
            }
            
            // Check again after a short delay
            setTimeout(() => pollJobStatus(jobId), 1000);
        } else if (data.status === 'completed') {
            hideLoading();
            
            if (data.images && data.images.length > 0) {
                displayGeneratedImages(data.images);
            }
            
            // Display logs if available
            if (data.logs) {
                displayLogs(data.logs);
            }
            
            showNotification('Generation completed successfully!', 'success');
        } else if (data.status === 'failed') {
            hideLoading();
            showNotification('Generation failed: ' + data.error, 'error');
        }
    } catch (error) {
        hideLoading();
        console.error('Error polling job status:', error);
        showNotification('Error checking generation status', 'error');
    }
}

function displayGeneratedImages(images) {
    const gallery = document.getElementById('generation-results');
    if (!gallery) return;
    
    // Clear existing contents
    gallery.innerHTML = '';
    
    // Add each image to the gallery
    images.forEach(img => {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'gallery-item';
        
        const imgElement = document.createElement('img');
        imgElement.src = `data:image/png;base64,${img.base64}`;
        imgElement.alt = img.filename;
        imgElement.dataset.path = img.path;
        
        imgElement.addEventListener('click', function() {
            openImageModal(this.src, img.filename, img.path);
        });
        
        const overlay = document.createElement('div');
        overlay.className = 'gallery-item-overlay';
        overlay.textContent = img.filename;
        
        imgContainer.appendChild(imgElement);
        imgContainer.appendChild(overlay);
        gallery.appendChild(imgContainer);
    });
}

function displayLogs(logs) {
    const logsContainer = document.getElementById('generation-logs');
    if (!logsContainer) return;
    
    if (Array.isArray(logs)) {
        logsContainer.innerHTML = logs.map(log => 
            `<div class="log-entry">${log}</div>`
        ).join('');
    } else {
        logsContainer.innerHTML = `<div class="log-entry">${logs}</div>`;
    }
    
    // Scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

// UI Helpers
function setupModalHandlers() {
    // Get the modal
    const modal = document.getElementById('modal');
    if (!modal) return;
    
    // Get the <span> element that closes the modal
    const closeBtn = modal.querySelector('.close-modal');
    
    // When the user clicks on <span> (x), close the modal
    closeBtn.addEventListener('click', function() {
        modal.classList.remove('show');
    });
    
    // When the user clicks anywhere outside of the modal content, close it
    window.addEventListener('click', function(event) {
        if (event.target === modal) {
            modal.classList.remove('show');
        }
    });
}

function openImageModal(src, filename, path) {
    const modal = document.getElementById('modal');
    const modalContent = document.getElementById('modal-content');
    
    if (!modal || !modalContent) return;
    
    // Set the modal content
    modalContent.innerHTML = `
        <div class="modal-header">
            <h3>${filename}</h3>
        </div>
        <div class="modal-body" style="text-align: center; padding: 20px;">
            <img src="${src}" alt="${filename}" style="max-width: 100%; max-height: 70vh;">
        </div>
        <div class="modal-footer" style="padding: 15px; display: flex; justify-content: space-between;">
            <div>
                <span>Path: ${path}</span>
            </div>
            <div>
                <a href="${src}" download="${filename}" class="btn btn-primary">Download</a>
            </div>
        </div>
    `;
    
    // Show the modal
    modal.classList.add('show');
}

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

function updateLoadingProgress(progress) {
    const loadingMessage = document.getElementById('loading-message');
    
    if (loadingMessage) {
        loadingMessage.textContent = `Processing... ${progress}%`;
    }
}

function showNotification(message, type = 'info') {
    // Create notification if it doesn't exist
    let notification = document.getElementById('notification');
    
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 5px;
            color: white;
            font-weight: 500;
            z-index: 1000;
            transition: all 0.3s ease;
            opacity: 0;
            transform: translateY(-20px);
        `;
        document.body.appendChild(notification);
    }
    
    // Set notification color based on type
    let bgColor;
    switch (type) {
        case 'success':
            bgColor = 'var(--success-color)';
            break;
        case 'error':
            bgColor = 'var(--error-color)';
            break;
        case 'warning':
            bgColor = 'var(--warning-color)';
            break;
        default:
            bgColor = 'var(--primary-color)';
    }
    
    notification.style.backgroundColor = bgColor;
    notification.textContent = message;
    
    // Show notification
    notification.style.opacity = '1';
    notification.style.transform = 'translateY(0)';
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.style.opacity = '0';
        notification.style.transform = 'translateY(-20px)';
    }, 3000);
}

// Token counting
async function countTokens(prompt) {
    try {
        const response = await fetch('/api/tokens', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ prompt })
        });
        
        const data = await response.json();
        return data.token_count;
    } catch (error) {
        console.error('Error counting tokens:', error);
        return 'Error';
    }
}

// Setup form handlers if present
if (document.getElementById('txt2img-form')) {
    setupTxt2ImgForm();
}

if (document.getElementById('img2img-form')) {
    setupImg2ImgForm();
}

if (document.getElementById('settings-form')) {
    setupSettingsForm();
}

if (document.getElementById('controlnet-form')) {
    setupControlNetForm();
}

function setupTxt2ImgForm() {
    const form = document.getElementById('txt2img-form');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        generateImage(formData, false);
    });
    
    // Token counting for prompt
    const promptField = form.querySelector('#prompt');
    const promptTokens = document.getElementById('prompt-tokens');
    
    if (promptField && promptTokens) {
        promptField.addEventListener('input', async function() {
            const count = await countTokens(this.value);
            promptTokens.textContent = count;
        });
    }
    
    // Token counting for negative prompt
    const negPromptField = form.querySelector('#negative-prompt');
    const negPromptTokens = document.getElementById('negative-prompt-tokens');
    
    if (negPromptField && negPromptTokens) {
        negPromptField.addEventListener('input', async function() {
            const count = await countTokens(this.value);
            negPromptTokens.textContent = count;
        });
    }
}

function setupImg2ImgForm() {
    const form = document.getElementById('img2img-form');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        generateImage(formData, true);
    });
    
    // Image preview
    const imageInput = form.querySelector('#init-image');
    const imagePreview = document.getElementById('init-image-preview');
    
    if (imageInput && imagePreview) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    imagePreview.src = event.target.result;
                    imagePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    // Token counting for prompt
    const promptField = form.querySelector('#prompt');
    const promptTokens = document.getElementById('prompt-tokens');
    
    if (promptField && promptTokens) {
        promptField.addEventListener('input', async function() {
            const count = await countTokens(this.value);
            promptTokens.textContent = count;
        });
    }
    
    // Token counting for negative prompt
    const negPromptField = form.querySelector('#negative-prompt');
    const negPromptTokens = document.getElementById('negative-prompt-tokens');
    
    if (negPromptField && negPromptTokens) {
        negPromptField.addEventListener('input', async function() {
            const count = await countTokens(this.value);
            negPromptTokens.textContent = count;
        });
    }
}

function setupSettingsForm() {
    const form = document.getElementById('settings-form');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(form);
        const settings = {};
        
        for (const [key, value] of formData.entries()) {
            settings[key] = value;
        }
        
        // Handle checkboxes
        const watermarkEnabled = document.getElementById('watermark-enabled');
        if (watermarkEnabled) {
            settings.watermark_enabled = watermarkEnabled.checked;
        }
        
        saveSettings(settings);
    });
}

function setupControlNetForm() {
    const form = document.getElementById('controlnet-form');
    
    // Image preview and processing
    const imageInput = form.querySelector('#control-image');
    const imagePreview = document.getElementById('control-image-preview');
    const processedPreview = document.getElementById('processed-image-preview');
    const processBtn = document.getElementById('process-btn');
    
    if (imageInput && imagePreview) {
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    imagePreview.src = event.target.result;
                    imagePreview.style.display = 'block';
                    
                    // Hide processed preview when new image is selected
                    if (processedPreview) {
                        processedPreview.style.display = 'none';
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }
    
    if (processBtn && imageInput) {
        processBtn.addEventListener('click', function() {
            if (!imageInput.files.length) {
                showNotification('Please select an image first', 'warning');
                return;
            }
            
            const formData = new FormData();
            formData.append('control_image', imageInput.files[0]);
            
            const controlnetType = document.getElementById('controlnet-type');
            if (controlnetType) {
                formData.append('type', controlnetType.value);
            }
            
            showLoading('Processing control image...');
            
            fetch('/api/process_controlnet', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                
                if (data.success && data.processed_image) {
                    processedPreview.src = `data:image/png;base64,${data.processed_image}`;
                    processedPreview.style.display = 'block';
                    
                    // Store processed image ID for generation
                    document.getElementById('processed-image-id').value = data.image_id;
                } else {
                    showNotification(data.error || 'Failed to process image', 'error');
                }
            })
            .catch(error => {
                hideLoading();
                console.error('Error processing control image:', error);
                showNotification('Error processing control image', 'error');
            });
        });
    }
} 