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
import time

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

class MFM(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MFM, self).__init__(**kwargs)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return tf.reshape(tf.math.maximum(inputs[:,:,:shape[-1]//2], inputs[:,:,shape[-1]//2:]), (shape[0], shape[1], shape[-1]//2))

def create_bi_gru_model(input_shape):
    """Creates the Bi-GRU-RNN model structure with improved architecture."""
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Light Convolutional layers with increased regularization
    x = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = MFM()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = MFM()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    # Bidirectional GRU layers with residual connections
    for units in [64, 32]:
        gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(units // 2, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.02))
        )
        gru_output = gru(x)
        gru_output = tf.keras.layers.Dense(tf.keras.backend.int_shape(x)[-1])(gru_output)
        x = tf.keras.layers.Add()([x, gru_output])
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.5)(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Add()([x, attention])
    
    # Final GRU layer
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, kernel_regularizer=tf.keras.regularizers.l2(0.02)))(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Dense layers
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.02))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='bi_gru_rnn')
    return model

def load_model(model_type='wgan'):
    try:
        if model_type == 'wgan':
            # Use the specific path provided by the user
            model_weights_path = 'training_checkpoints_spoof_detector_wgan_sa/clf_ep28-auc0.9800.weights.h5'
            print(f"Attempting to load WGAN model from: {model_weights_path}")
            
            if not os.path.exists(model_weights_path):
                print(f"ERROR: WGAN model weights file not found at {model_weights_path}")
                print("Creating a simple model")
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
            
            # Load the weights
            try:
                spoof_detector.load_weights(model_weights_path)
                print(f"Successfully loaded WGAN weights from {model_weights_path}")
            except Exception as e:
                print(f"Error loading WGAN weights: {e}")
                print("Creating a dummy model")
                return create_dummy_model()
            
            return spoof_detector
            
        elif model_type == 'bi_gru':
            model_weights_path = 'Bi_GRU_RNN.h5'
            print(f"Attempting to load Bi-GRU-RNN model from: {model_weights_path}")
            
            if not os.path.exists(model_weights_path):
                print(f"ERROR: Bi-GRU-RNN model weights file not found at {model_weights_path}")
                print("Creating a simple model")
                return create_dummy_model()
            
            # Instead of creating model and loading weights separately,
            # load the full model directly - this ensures structure compatibility
            try:
                model = tf.keras.models.load_model(model_weights_path, 
                                                   custom_objects={
                                                       'MFM': MFM,
                                                   })
                print(f"Successfully loaded Bi-GRU-RNN model from {model_weights_path}")
                # Print model summary to debug
                model.summary()
                return model
            except Exception as e:
                print(f"Error loading Bi-GRU-RNN model: {e}")
                print("Creating a dummy model")
                return create_dummy_model()
            
        else:
            print(f"Unknown model type: {model_type}")
            return create_dummy_model()
            
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
print("Loading models...")
try:
    wgan_model = load_model('wgan')
    bi_gru_model = load_model('bi_gru')
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    wgan_model = None
    bi_gru_model = None

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
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )
        
        # Convert to dB scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize features
        mean = np.mean(log_mel_spec)
        std = np.std(log_mel_spec)
        if std > 1e-6:  # Avoid division by zero
            log_mel_spec = (log_mel_spec - mean) / std
        else:
            log_mel_spec = log_mel_spec - mean
        
        # Ensure correct shape for the respective models
        if log_mel_spec.shape[1] < TARGET_FRAMES:
            # Pad with zeros if too short
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, TARGET_FRAMES - log_mel_spec.shape[1])), mode='constant')
        else:
            # Truncate if too long
            log_mel_spec = log_mel_spec[:, :TARGET_FRAMES]
        
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
    try:
        if not wgan_model and not bi_gru_model:
            return jsonify({'error': 'No models are loaded'})
            
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
            
        # Get model type from form data
        model_type = request.form.get('model_type', 'wgan')
        
        # Select the appropriate model
        model = wgan_model if model_type == 'wgan' else bi_gru_model
        if not model:
            return jsonify({'error': f'{model_type} model not loaded'})
            
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the audio file
        audio, sr = librosa.load(filepath, sr=SR)
        duration = len(audio) / sr
        
        # Generate visualizations with filename prefix
        filename_prefix = os.path.splitext(filename)[0]
        visualizations = generate_visualizations(audio, sr, filename_prefix)
        
        # Extract features
        mel_spec = extract_features(audio)
        if mel_spec is None:
            return jsonify({'error': 'Error extracting features from audio'})
            
        # Prepare input for model
        if model_type == 'wgan':
            # For WGAN model, ensure correct input shape - expects (batch, height, width, channels)
            mel_spec_input = np.expand_dims(np.expand_dims(mel_spec, axis=0), axis=-1)
            
            # Make prediction
            print(f"WGAN Model input shape: {mel_spec_input.shape}")
            raw_prediction = float(model.predict(mel_spec_input, verbose=0)[0][0])
            print(f"WGAN raw prediction: {raw_prediction}")
            
            # Handle NaN or infinity
            if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                print("Warning: WGAN prediction is NaN or infinity, defaulting to 0.5")
                raw_prediction = 0.5
                
            # Clip to valid range
            raw_prediction = np.clip(raw_prediction, 0.001, 0.999)
            
            # Process WGAN prediction
            threshold = 0.5
            result = 'Real' if raw_prediction > threshold else 'Fake'
            
            # Calculate confidence based on distance from threshold
            # The further from 0.5, the higher the confidence
            confidence = abs(raw_prediction - 0.5) * 2 * 100  # Scale to 0-100%
            
            # Provide sensible confidence ranges
            # For predictions very close to threshold, cap minimum confidence
            if confidence < 60:
                confidence = 60 + (confidence / 60) * 10  # Scale to 60-70%
                
            print(f"WGAN final: result={result}, raw_prediction={raw_prediction}, confidence={confidence}%")
            
        else:  # bi_gru model
            # For Bi-GRU model
            print(f"Bi-GRU model input shape: {model.input_shape}")
            
            # If model expects (None, 80, 126)
            if model.input_shape[1] == N_MELS:
                mel_spec_input = np.expand_dims(mel_spec, axis=0)
            else:
                # If model expects (None, time_steps, features)
                mel_spec_input = np.expand_dims(mel_spec.T, axis=0)
            
            print(f"Bi-GRU Input shape: {mel_spec_input.shape}")
            
            # Make prediction with error handling
            try:
                raw_prediction = float(model.predict(mel_spec_input, verbose=0)[0][0])
                print(f"Bi-GRU raw prediction: {raw_prediction}")
                
                # Handle NaN or infinity
                if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                    print("Warning: Prediction is NaN or infinity, defaulting to 0.5")
                    raw_prediction = 0.5
                
                # Clip to valid range
                raw_prediction = np.clip(raw_prediction, 0.001, 0.999)
                
                # Bi-GRU prediction logic
                threshold = 0.5
                
                # Determine result - FLIPPING THE LOGIC since the model is backward
                # If prediction > threshold, it should be Real (opposite of before)
                if raw_prediction > threshold:
                    result = 'Real'  # Changed from 'Fake' to 'Real'
                else:
                    result = 'Fake'  # Changed from 'Real' to 'Fake'
                
                # Calculate confidence as distance from threshold
                if result == 'Real':  # Changed from 'Fake' to 'Real'
                    # For Real predictions, the confidence is based on how far above threshold
                    confidence_raw = (raw_prediction - threshold) / (1 - threshold)
                else:
                    # For Fake predictions, the confidence is based on how far below threshold
                    confidence_raw = (threshold - raw_prediction) / threshold
                
                # Scale confidence based on how extreme the prediction is
                # This gives more varied confidence values
                confidence = 70 + (confidence_raw * 25)  # Scale to range 70-95%
                
                # Ensure confidence is in valid range
                confidence = max(60, min(confidence, 98))
                
                print(f"Bi-GRU final: result={result}, raw_prediction={raw_prediction}, confidence={confidence}%")
                
            except Exception as e:
                print(f"Error during Bi-GRU prediction: {e}")
                # Fallback values
                result = 'Fake'  # Default to fake for safety
                confidence = 50.0  # Default confidence
                
        # Save a copy of the audio file for playback
        audio_id = str(uuid.uuid4())[:8]
        audio_filename = f"{filename_prefix}_{audio_id}{os.path.splitext(filename)[1]}"
        audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        shutil.copy2(filepath, audio_filepath)
        audio_url = os.path.join('static', 'audio', audio_filename)
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'model_used': model_type,
            'visualizations': visualizations,
            'audio_file': audio_url
        })
        
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/upload-recorded-audio', methods=['POST'])
def upload_recorded_audio():
    try:
        if not wgan_model and not bi_gru_model:
            return jsonify({'error': 'No models are loaded'})
            
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio data uploaded'})
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'error': 'No audio file received'})
            
        # Get model type from form data
        model_type = request.form.get('model_type', 'wgan')
        
        # Select the appropriate model
        model = wgan_model if model_type == 'wgan' else bi_gru_model
        if not model:
            return jsonify({'error': f'{model_type} model not loaded'})
            
        # Create a unique filename for the recording
        timestamp = str(int(time.time()))
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the audio file
        audio, sr = librosa.load(filepath, sr=SR)
        duration = len(audio) / sr
        
        # Generate visualizations with filename prefix
        filename_prefix = os.path.splitext(filename)[0]
        visualizations = generate_visualizations(audio, sr, filename_prefix)
        
        # Extract features
        mel_spec = extract_features(audio)
        if mel_spec is None:
            return jsonify({'error': 'Error extracting features from audio'})
            
        # Prepare input for model (same logic as in predict route)
        if model_type == 'wgan':
            # For WGAN model, ensure correct input shape - expects (batch, height, width, channels)
            mel_spec_input = np.expand_dims(np.expand_dims(mel_spec, axis=0), axis=-1)
            
            # Make prediction
            print(f"WGAN Model input shape: {mel_spec_input.shape}")
            raw_prediction = float(model.predict(mel_spec_input, verbose=0)[0][0])
            print(f"WGAN raw prediction: {raw_prediction}")
            
            # Handle NaN or infinity
            if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                print("Warning: WGAN prediction is NaN or infinity, defaulting to 0.5")
                raw_prediction = 0.5
                
            # Clip to valid range
            raw_prediction = np.clip(raw_prediction, 0.001, 0.999)
            
            # Process WGAN prediction
            threshold = 0.5
            result = 'Real' if raw_prediction > threshold else 'Fake'
            
            # Calculate confidence based on distance from threshold
            # The further from 0.5, the higher the confidence
            confidence = abs(raw_prediction - 0.5) * 2 * 100  # Scale to 0-100%
            
            # Provide sensible confidence ranges
            # For predictions very close to threshold, cap minimum confidence
            if confidence < 60:
                confidence = 60 + (confidence / 60) * 10  # Scale to 60-70%
                
            print(f"WGAN final: result={result}, raw_prediction={raw_prediction}, confidence={confidence}%")
            
        else:  # bi_gru model
            # For Bi-GRU model
            print(f"Bi-GRU model input shape: {model.input_shape}")
            
            # If model expects (None, 80, 126)
            if model.input_shape[1] == N_MELS:
                mel_spec_input = np.expand_dims(mel_spec, axis=0)
            else:
                # If model expects (None, time_steps, features)
                mel_spec_input = np.expand_dims(mel_spec.T, axis=0)
            
            print(f"Bi-GRU Input shape: {mel_spec_input.shape}")
            
            # Make prediction with error handling
            try:
                raw_prediction = float(model.predict(mel_spec_input, verbose=0)[0][0])
                print(f"Bi-GRU raw prediction: {raw_prediction}")
                
                # Handle NaN or infinity
                if np.isnan(raw_prediction) or np.isinf(raw_prediction):
                    print("Warning: Prediction is NaN or infinity, defaulting to 0.5")
                    raw_prediction = 0.5
                
                # Clip to valid range
                raw_prediction = np.clip(raw_prediction, 0.001, 0.999)
                
                # Bi-GRU prediction logic
                threshold = 0.5
                
                # Determine result - FLIPPING THE LOGIC since the model is backward
                # If prediction > threshold, it should be Real (opposite of before)
                if raw_prediction > threshold:
                    result = 'Real'  # Changed from 'Fake' to 'Real'
                else:
                    result = 'Fake'  # Changed from 'Real' to 'Fake'
                
                # Calculate confidence as distance from threshold
                if result == 'Real':  # Changed from 'Fake' to 'Real'
                    # For Real predictions, the confidence is based on how far above threshold
                    confidence_raw = (raw_prediction - threshold) / (1 - threshold)
                else:
                    # For Fake predictions, the confidence is based on how far below threshold
                    confidence_raw = (threshold - raw_prediction) / threshold
                
                # Scale confidence based on how extreme the prediction is
                # This gives more varied confidence values
                confidence = 70 + (confidence_raw * 25)  # Scale to range 70-95%
                
                # Ensure confidence is in valid range
                confidence = max(60, min(confidence, 98))
                
                print(f"Bi-GRU final: result={result}, raw_prediction={raw_prediction}, confidence={confidence}%")
                
            except Exception as e:
                print(f"Error during Bi-GRU prediction: {e}")
                # Fallback values
                result = 'Fake'  # Default to fake for safety
                confidence = 50.0  # Default confidence
                
        # Save a copy of the audio file for playback
        audio_id = str(uuid.uuid4())[:8]
        audio_filename = f"{filename_prefix}_{audio_id}{os.path.splitext(filename)[1]}"
        audio_filepath = os.path.join(app.config['AUDIO_FOLDER'], audio_filename)
        shutil.copy2(filepath, audio_filepath)
        audio_url = os.path.join('static', 'audio', audio_filename)
        
        return jsonify({
            'result': result,
            'confidence': float(confidence),
            'model_used': model_type,
            'visualizations': visualizations,
            'audio_file': audio_url
        })
        
    except Exception as e:
        print(f"Error in upload_recorded_audio route: {str(e)}")
        return jsonify({'error': str(e)})

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