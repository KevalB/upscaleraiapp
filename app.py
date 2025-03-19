from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from moviepy.editor import VideoFileClip
import threading
from queue import Queue
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
TEMP_FOLDER = 'static/temp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tif', 'tiff', 'heic', 
                     'mp4', 'webm', 'mkv', 'flv', 'gif', 'm4v', 'avi', 'mov', 'qt', '3gp', 'mpg', 'mpeg'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Global variables for progress tracking
progress_queue = Queue()
current_progress = 0

# Initialize ONNX runtime with CPU only
providers = ['CPUExecutionProvider']
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Load AI models
def load_model(model_name):
    # Get the absolute path to the AI-onnx directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    # Map model names to actual filenames
    model_files = {
        'RealESRGANx4': 'RealESRGANx4_fp16.onnx',
        'BSRGANx4': 'BSRGANx4.onnx',
        'BSRGANx2': 'BSRGANx2.onnx'
    }
    
    if model_name not in model_files:
        raise ValueError(f"Unknown model name: {model_name}")
        
    model_path = os.path.join(parent_dir, 'AI-onnx', model_files[model_name])
    
    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        session = ort.InferenceSession(model_path, providers=providers, sess_options=session_options)
        print(f"Successfully loaded model: {model_name}")
        return session
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise

# Initialize models
try:
    print("Starting model initialization...")
    realesrgan_model = load_model('RealESRGANx4')
    bsrganx4_model = load_model('BSRGANx4')
    bsrganx2_model = load_model('BSRGANx2')
    print("All models loaded successfully!")
except Exception as e:
    print(f"Error during model initialization: {str(e)}")
    realesrgan_model = None
    bsrganx4_model = None
    bsrganx2_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, scale_factor, interpolation, model_name):
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        height, width = img.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
        
        # Convert to float32 and normalize
        img = img.astype(np.float32) / 255.0
        
        # Transpose to NCHW format (batch, channels, height, width)
        img = np.transpose(img, (2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Select model
        if model_name == 'RealESRGANx4':
            model = realesrgan_model
        elif model_name == 'BSRGANx4':
            model = bsrganx4_model
        else:  # BSRGANx2
            model = bsrganx2_model
            
        if model is None:
            raise ValueError(f"Model {model_name} not loaded")
        
        # Run inference
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: img})[0]
        
        # Process output
        result = np.squeeze(result)
        result = np.transpose(result, (1, 2, 0))  # Convert back to HWC format
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # Convert back to BGR for saving
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

def process_video(video_path, scale_factor, interpolation, model_name):
    try:
        # Load video
        video = VideoFileClip(video_path)
        
        # Create output video path
        output_path = os.path.join(RESULT_FOLDER, f'upscaled_{os.path.basename(video_path)}')
        
        # Process frames
        def process_frame(frame):
            # Convert frame to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save frame temporarily
            temp_frame_path = os.path.join(TEMP_FOLDER, 'temp_frame.png')
            cv2.imwrite(temp_frame_path, frame)
            
            # Process frame
            processed_frame = process_image(temp_frame_path, scale_factor, interpolation, model_name)
            
            # Update progress
            global current_progress
            current_progress = min(100, current_progress + (100 / video.duration))
            progress_queue.put(current_progress)
            
            return processed_frame
        
        # Process video
        processed_video = video.fl_image(process_frame)
        
        # Write output video
        processed_video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        
        # Clean up
        video.close()
        processed_video.close()
        
        return output_path
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Get parameters
        scale_factor = float(request.form.get('scale', 2.0))
        interpolation = int(request.form.get('interpolation', 50))
        model_name = request.form.get('model', 'RealESRGANx4')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process file based on type
        if file.filename.lower().endswith(('.mp4', '.webm', '.mkv', '.flv', '.gif', '.m4v', '.avi', '.mov', '.qt', '.3gp', '.mpg', '.mpeg')):
            # Reset progress
            global current_progress
            current_progress = 0
            
            # Process video in background
            def process_video_thread():
                try:
                    output_path = process_video(file_path, scale_factor, interpolation, model_name)
                    progress_queue.put(100)  # Signal completion
                except Exception as e:
                    progress_queue.put(-1)  # Signal error
                    print(f"Error in video processing thread: {str(e)}")
            
            thread = threading.Thread(target=process_video_thread)
            thread.start()
            
            return jsonify({
                'type': 'video',
                'message': 'Video processing started'
            })
            
        else:
            # Process image
            result = process_image(file_path, scale_factor, interpolation, model_name)
            
            # Save result
            output_path = os.path.join(RESULT_FOLDER, f'upscaled_{filename}')
            cv2.imwrite(output_path, result)
            
            return jsonify({
                'type': 'image',
                'result_url': f'/static/results/upscaled_{filename}'
            })
            
    except Exception as e:
        print(f"Error in upscale route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/progress')
def get_progress():
    try:
        while not progress_queue.empty():
            global current_progress
            current_progress = progress_queue.get()
        return jsonify({'progress': current_progress})
    except Exception as e:
        print(f"Error getting progress: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)