import os
# Force TensorFlow to use CPU and disable unnecessary warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom operations

# Set matplotlib to use non-interactive backend to avoid Tcl/Tk issues
import matplotlib
matplotlib.use('Agg')  # Must be before importing pyplot

# Completely disable GPU support in TensorFlow
import tensorflow as tf
try:
    # Physical devices have to be set before GPUs are initialized
    tf.config.set_visible_devices([], 'GPU')
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, False)
        tf.config.experimental.set_virtual_device_configuration(
            device,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=0)]
        )
except Exception as e:
    print(f"Error disabling GPU: {e}")

# Additional TensorFlow configuration
tf.get_logger().setLevel('ERROR')  # Only show TensorFlow errors

# Import other libraries
from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import librosa.display
import tensorflow as tf

from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import io
import base64
import uuid
import socket
import shutil
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Custom SelfAttention layer needed for the model
class SelfAttention(tf.keras.layers.Layer):
    """
    Self-attention layer based on SAGAN.
    Input shape: (batch, height, width, channels)
    Output shape: (batch, height, width, channels_out) where channels_out is typically channels
    """
    def __init__(self, channels_out=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels_out = channels_out

    def build(self, input_shape):
        self.input_channels = input_shape[-1]
        if self.channels_out is None:
            self.channels_out = self.input_channels

        # Convolution layers for query, key, value
        self.f = tf.keras.layers.Conv2D(self.input_channels // 8, kernel_size=1, strides=1, padding='same', name='conv_f') # Query
        self.g = tf.keras.layers.Conv2D(self.input_channels // 8, kernel_size=1, strides=1, padding='same', name='conv_g') # Key
        self.h = tf.keras.layers.Conv2D(self.channels_out, kernel_size=1, strides=1, padding='same', name='conv_h')        # Value

        # Final 1x1 convolution
        self.out_conv = tf.keras.layers.Conv2D(self.channels_out, kernel_size=1, strides=1, padding='same', name='conv_out')

        # Learnable scale parameter
        self.gamma = self.add_weight(name='gamma', shape=(1,), initializer='zeros', trainable=True)

        super(SelfAttention, self).build(input_shape)

    def call(self, x):
        batch_size, height, width, num_channels = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        location_num = height * width
        downsampled_num = location_num

        # Query (f), Key (g), Value (h) projections
        f_proj = self.f(x) # Shape: (batch, h, w, c/8)
        g_proj = self.g(x) # Shape: (batch, h, w, c/8)
        h_proj = self.h(x) # Shape: (batch, h, w, c_out)

        # Reshape for matrix multiplication
        f_flat = tf.reshape(f_proj, shape=(batch_size, location_num, self.input_channels // 8)) # (batch, h*w, c/8)
        g_flat = tf.reshape(g_proj, shape=(batch_size, location_num, self.input_channels // 8)) # (batch, h*w, c/8)
        h_flat = tf.reshape(h_proj, shape=(batch_size, location_num, self.channels_out))       # (batch, h*w, c_out)

        # Attention map calculation
        # Transpose g for matmul: (batch, c/8, h*w)
        g_flat_t = tf.transpose(g_flat, perm=[0, 2, 1])
        # Attention score: (batch, h*w, c/8) x (batch, c/8, h*w) -> (batch, h*w, h*w)
        attention_score = tf.matmul(f_flat, g_flat_t)
        attention_prob = tf.nn.softmax(attention_score, axis=-1) # Apply softmax across locations

        # Apply attention map to value projection
        # (batch, h*w, h*w) x (batch, h*w, c_out) -> (batch, h*w, c_out)
        attention_output = tf.matmul(attention_prob, h_flat)

        # Reshape back to image format
        attention_output_reshaped = tf.reshape(attention_output, shape=(batch_size, height, width, self.channels_out))

        # Apply final 1x1 convolution and scale by gamma
        o = self.out_conv(attention_output_reshaped)
        y = self.gamma * o + x # Additive skip connection

        return y

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({"channels_out": self.channels_out})
        return config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['STATIC_FOLDER'] = 'static'
app.config['VISUALIZATIONS_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'visualizations')
app.config['AUDIO_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'audio')

# Ensure required directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VISUALIZATIONS_FOLDER'], exist_ok=True)
os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)

# Model parameters
SR = 16000
DURATION = 4
N_MELS = 80
N_FFT = 2048
HOP_LENGTH = 512
TARGET_FRAMES = 126

# Function to create critic model (needed for model structure)
def create_critic(input_shape):
    """Creates the Critic model with Self-Attention. NO DROPOUT."""
    model_input_shape = (input_shape[0], input_shape[1], 1)  # Expects (80, 126, 1)

    model = tf.keras.models.Sequential(name='critic')
    model.add(tf.keras.layers.Input(shape=model_input_shape))

    # Layer 1
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # Layer 2
    model.add(tf.keras.layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # Add Self-Attention Layer
    model.add(SelfAttention(channels_out=128))

    # Layer 3
    model.add(tf.keras.layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.LayerNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.2))

    # Flatten and Output Score
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))

    return model

# Load the model
def load_model():
    try:
        # Use the specific path provided by the user
        model_weights_path = 'training_checkpoints_spoof_detector_wgan_sa/clf_ep28-auc0.9800.weights.h5'
        
        print(f"Attempting to load model from: {model_weights_path}")
        
        if not os.path.exists(model_weights_path):
            print(f"ERROR: Model weights file not found at {model_weights_path}")
            print("Checking if directory exists...")
            
            dir_path = os.path.dirname(model_weights_path)
            if not os.path.exists(dir_path):
                print(f"Directory {dir_path} does not exist")
                # Try to create the directory
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory {dir_path}")
                except Exception as e:
                    print(f"Error creating directory: {e}")
            else:
                print(f"Directory {dir_path} exists, but file not found")
                # List files in the directory
                try:
                    files = os.listdir(dir_path)
                    print(f"Files in {dir_path}: {files}")
                except Exception as e:
                    print(f"Error listing directory: {e}")
            
            # If file doesn't exist, try alternative weights
            print("Trying alternative weights...")
            for alt_file in [
                'spoof_detector_ep200_finetuned_best_val_auc.weights.h5',
                'spoof_detector_best_val_auc_wgan_sa.weights.h5',
                'spoof_detector_final_wgan_sa.weights.h5',
                'spoof_detector_ep180_finetuned_best_val_loss.weights.h5'
            ]:
                if os.path.exists(alt_file):
                    print(f"Found alternative weights: {alt_file}")
                    model_weights_path = alt_file
                    break
            else:
                print("No alternative weights found, creating a simple model")
                return create_dummy_model()
        
        # Recreate model structure for classification
        critic_base = create_critic((N_MELS, TARGET_FRAMES))
        spoof_detector = tf.keras.models.Sequential(name='spoof_detector')
        
        # Create classifier model structure
        spoof_detector.add(tf.keras.layers.Input(shape=(N_MELS, TARGET_FRAMES, 1)))
        
        # Add all layers from critic except the last one
        for layer in critic_base.layers[:-1]:
            spoof_detector.add(layer)
        
        # Add final sigmoid layer for binary classification
        spoof_detector.add(tf.keras.layers.Dense(1, activation='sigmoid', name='classifier_output'))
        
        # Build the model with a dummy input
        dummy_input = tf.zeros((1, N_MELS, TARGET_FRAMES, 1), dtype=tf.float32)
        _ = spoof_detector(dummy_input, training=False)
        
        # Load the weights
        try:
            spoof_detector.load_weights(model_weights_path)
            print(f"Successfully loaded weights from {model_weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Creating a dummy model")
            return create_dummy_model()
        
        # Display model architecture
        print("Model architecture summary:")
        spoof_detector.summary()
        
        return spoof_detector
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        
        print("Creating a simple model as fallback")
        return create_dummy_model()

def create_dummy_model():
    """Creates a simple CNN model as a fallback when the main model fails to load."""
    print("Initializing simplified dummy model")
    try:
        # Create a simple Sequential model
        model = tf.keras.models.Sequential(name='simple_spoof_detector')
        model.add(tf.keras.layers.Input(shape=(N_MELS, TARGET_FRAMES, 1)))
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), padding='same'))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        
        # Compile the model to ensure it's ready for inference
        model.compile(optimizer='adam', 
                    loss='binary_crossentropy', 
                    metrics=['accuracy'])
        
        # Build the model with a dummy input to ensure it's initialized
        dummy_input = tf.zeros((1, N_MELS, TARGET_FRAMES, 1))
        _ = model(dummy_input, training=False)
        
        print("Dummy model created successfully")
        return model
    except Exception as e:
        print(f"Error creating dummy model: {e}")
        # Create an absolute fallback model using Functional API
        print("Creating fallback functional model")
        inputs = tf.keras.layers.Input(shape=(N_MELS, TARGET_FRAMES, 1))
        x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs, name='fallback_model')
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

# Load model at startup
print("Loading model...")
try:
    model = load_model()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def load_and_preprocess_audio(file_path, sr=SR, duration=DURATION):
    try:
        audio, current_sr = librosa.load(file_path, sr=None, duration=duration)
        if current_sr != sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
        
        target_len = sr * duration
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            audio = audio[:target_len]
            
        max_amp = np.max(np.abs(audio))
        if max_amp > 1e-6:
            audio = audio / max_amp
            
        return audio
    except Exception as e:
        print(f"Error loading/preprocessing {file_path}: {e}")
        return None

def extract_features(audio, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    if audio is None:
        return None
    try:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec)
        if std > 1e-6:
            log_mel_spec = (log_mel_spec - mean) / std
        else:
            log_mel_spec = log_mel_spec - mean
            
        return log_mel_spec
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def generate_visualizations(audio, sr, filename_prefix):
    """Generate different visualizations of audio data."""
    visualizations = {}
    
    # Create a unique ID for this audio visualization set
    viz_id = str(uuid.uuid4())[:8]
    
    try:
        # Use non-interactive backend to avoid Tk issues
        plt.switch_backend('Agg')
        
        # 1. Waveform
        plt.figure(figsize=(10, 4))
        plt.title('Waveform')
        librosa.display.waveshow(audio, sr=sr)
        plt.tight_layout()
        wave_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename_prefix}_waveform_{viz_id}.png')
        plt.savefig(wave_path)
        plt.close('all')  # Explicitly close all figures
        visualizations['waveform'] = os.path.join('static', 'visualizations', os.path.basename(wave_path))
        
        # 2. Mel Spectrogram
        plt.figure(figsize=(10, 4))
        plt.title('Mel Spectrogram')
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, hop_length=HOP_LENGTH)
        plt.colorbar(img, format='%+2.0f dB')
        plt.tight_layout()
        mel_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename_prefix}_melspec_{viz_id}.png')
        plt.savefig(mel_path)
        plt.close('all')  # Explicitly close all figures
        visualizations['mel_spectrogram'] = os.path.join('static', 'visualizations', os.path.basename(mel_path))
        
        # 3. Chromagram
        plt.figure(figsize=(10, 4))
        plt.title('Chromagram')
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
        plt.colorbar(img)
        plt.tight_layout()
        chroma_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename_prefix}_chroma_{viz_id}.png')
        plt.savefig(chroma_path)
        plt.close('all')  # Explicitly close all figures
        visualizations['chromagram'] = os.path.join('static', 'visualizations', os.path.basename(chroma_path))
        
        # 4. Spectral Contrast
        plt.figure(figsize=(10, 4))
        plt.title('Spectral Contrast')
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        img = librosa.display.specshow(contrast, x_axis='time')
        plt.colorbar(img)
        plt.tight_layout()
        contrast_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename_prefix}_contrast_{viz_id}.png')
        plt.savefig(contrast_path)
        plt.close('all')  # Explicitly close all figures
        visualizations['spectral_contrast'] = os.path.join('static', 'visualizations', os.path.basename(contrast_path))
        
        # 5. MFCC
        plt.figure(figsize=(10, 4))
        plt.title('MFCC')
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        img = librosa.display.specshow(mfcc, x_axis='time')
        plt.colorbar(img)
        plt.tight_layout()
        mfcc_path = os.path.join(app.config['VISUALIZATIONS_FOLDER'], f'{filename_prefix}_mfcc_{viz_id}.png')
        plt.savefig(mfcc_path)
        plt.close('all')  # Explicitly close all figures
        visualizations['mfcc'] = os.path.join('static', 'visualizations', os.path.basename(mfcc_path))
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Ensure all matplotlib resources are released
    plt.close('all')
    
    return visualizations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. The application could not find or load a compatible model weights file. Please check the server logs for details.'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    filename = file.filename.lower()
    print(f"Received file: {file.filename}, Content-Type: {file.content_type}")
    
    try:
        # Process the file regardless of extension
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Save a copy for playback
        audio_id = str(uuid.uuid4())[:8]
        audio_filename = f"{os.path.splitext(filename)[0]}_{audio_id}{os.path.splitext(filename)[1]}"
        audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        shutil.copy2(filepath, audio_filepath)
        audio_url = os.path.join('static', 'audio', audio_filename)
        
        # Process audio
        audio = load_and_preprocess_audio(filepath)
        if audio is None:
            return jsonify({'error': 'Error processing audio file. The file may be corrupted or in an unsupported format.'}), 400
        
        # Initialize visualizations to empty
        visualizations = {}
        
        # Check if visualizations are requested (always true for now, but could be made optional)
        generate_viz = True
        if generate_viz:
            try:
                # Generate visualizations
                viz_prefix = os.path.splitext(filename)[0]
                visualizations = generate_visualizations(audio, SR, viz_prefix)
            except Exception as e:
                print(f"Error generating visualizations (continuing without them): {e}")
                import traceback
                traceback.print_exc()
                # Continue without visualizations if they fail
                visualizations = {}
        
        try:
            features = extract_features(audio)
            if features is None:
                return jsonify({'error': 'Error extracting features from audio. The file may not contain proper audio content.'}), 400
                
            # Ensure correct shape
            if features.shape[1] != TARGET_FRAMES:
                if features.shape[1] < TARGET_FRAMES:
                    features = np.pad(features, ((0, 0), (0, TARGET_FRAMES - features.shape[1])), mode='constant')
                else:
                    features = features[:, :TARGET_FRAMES]
                    
            # Add channel dimension and batch dimension
            features = np.expand_dims(features, axis=-1)
            features = np.expand_dims(features, axis=0)
            
            # Make prediction - ensure we're using eager execution compatible approach
            try:
                # Direct prediction in eager mode
                prediction = model(features, training=False).numpy()[0][0]
            except Exception as e:
                print(f"Error with eager execution prediction: {e}")
                # Fallback to older style predict method
                prediction = model.predict(features, verbose=0)[0][0]
                
            # Flip the prediction logic - now predictions < 0.5 are deepfake, >= 0.5 are real
            result = 'Deepfake' if prediction < 0.5 else 'Real'
            confidence = 1 - prediction if prediction < 0.5 else prediction
            
            # Select a random GIF from the appropriate reactions folder
            gif_url = None
            try:
                reaction_folder = "Real" if result == "Real" else "Fake"
                reaction_path = os.path.join("Reactions", reaction_folder)
                
                if os.path.exists(reaction_path):
                    reaction_files = [f for f in os.listdir(reaction_path) if f.lower().endswith('.gif')]
                    if reaction_files:
                        chosen_gif = random.choice(reaction_files)
                        reaction_gif = os.path.join(reaction_path, chosen_gif)
                        
                        # Create a copy in the static folder for the web server to access
                        static_gif_folder = os.path.join(app.config['STATIC_FOLDER'], 'gifs')
                        os.makedirs(static_gif_folder, exist_ok=True)
                        
                        gif_filename = f"{result.lower()}_{audio_id}.gif"
                        static_gif_path = os.path.join(static_gif_folder, gif_filename)
                        shutil.copy2(reaction_gif, static_gif_path)
                        
                        gif_url = os.path.join('static', 'gifs', gif_filename)
            except Exception as e:
                print(f"Error selecting reaction GIF (continuing without it): {e}")
                import traceback
                traceback.print_exc()
                # Continue without GIF if it fails
                gif_url = None
                        
            # Clean up original upload but keep the copy for playback
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error removing temporary file: {e}")
            
            return jsonify({
                'result': result,
                'confidence': float(confidence),
                'raw_score': float(prediction),
                'visualizations': visualizations,
                'audio_file': audio_url,
                'reaction_gif': gif_url
            })
        except Exception as e:
            print(f"Error during feature extraction or prediction: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        print("=" * 50)
        print("STARTING FLASK SERVER")
        print("=" * 50)
        
        # Initialize directories
        for directory in [app.config['UPLOAD_FOLDER'], app.config['VISUALIZATIONS_FOLDER'], app.config['AUDIO_FOLDER']]:
            if not os.path.exists(directory):
                print(f"Creating directory: {directory}")
                os.makedirs(directory, exist_ok=True)
        
        # Try multiple ports if one is busy
        ports = [8888, 8000, 5000, 9000]
        host = '127.0.0.1'
        
        for port in ports:
            try:
                print(f"Attempting to start server on http://{host}:{port}")
                print("=" * 50)
                app.run(host=host, port=port, debug=False)
                break  # If successful, exit the loop
            except OSError as e:
                if "Address already in use" in str(e):
                    print(f"Port {port} is already in use, trying another port...")
                else:
                    print(f"Error starting server on port {port}: {e}")
                    raise  # Re-raise if it's not a port-in-use error
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        import traceback
        traceback.print_exc()
        print("\nTry running manually with:")
        print("export FLASK_APP=app.py")
        print("flask run --host=127.0.0.1 --port=9000") 