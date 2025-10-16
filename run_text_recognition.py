import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Import GPU configuration
from gpu_config import gpu_config, configure_gpu_for_tensorflow

# Set Keras backend
os.environ["KERAS_BACKEND"] = "tensorflow"

# Configure GPU before importing TensorFlow
gpu_config.set_environment_variables()
gpu_config.optimize_for_training()

import numpy as np
import tensorflow as tf
import keras
from keras import ops
from keras import layers
from keras import models
from pathlib import Path
import json

# Configure TensorFlow for GPU usage
gpu_enabled = configure_gpu_for_tensorflow()

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"GPU Configuration: {'Enabled' if gpu_enabled else 'Disabled (using CPU)'}")
print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")

# Configuration
class DatasetConfig:
    def __init__(self):
        self.data_path = Path("./captcha_images_v2/")
        self.batch_size = 16
        self.image_width = 200
        self.image_height = 50
        self.pool_factor = 4
        self.train_split = 0.9
        self.validation_split = 0.1

config = DatasetConfig()

# Check if dataset exists
if not config.data_path.exists():
    print("Dataset not found. Please download it first.")
    sys.exit(1)

# Load dataset
image_files = sorted(list(map(str, list(config.data_path.glob("*.png")))))
text_labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in image_files]

# Extract unique character set
unique_chars = set(char for text in text_labels for char in text)
unique_chars = sorted(list(unique_chars))

print(f"\nDataset Statistics:")
print(f"==================")
print(f"Total samples: {len(image_files)}")
print(f"Unique characters: {len(unique_chars)}")
print(f"Character set: {unique_chars}")
print(f"Max sequence length: {max([len(text) for text in text_labels])}")
print(f"Min sequence length: {min([len(text) for text in text_labels])}")

# Character Encoder
class CharacterEncoder:
    def __init__(self, characters):
        self.char_to_int = layers.StringLookup(
            vocabulary=list(characters), 
            mask_token=None
        )
        self.int_to_char = layers.StringLookup(
            vocabulary=self.char_to_int.get_vocabulary(), 
            mask_token=None, 
            invert=True
        )
        self.vocab_size = len(self.char_to_int.get_vocabulary()) + 1
    
    def encode(self, text):
        return self.char_to_int(tf.strings.unicode_split(text, input_encoding="UTF-8"))
    
    def decode(self, integers):
        return tf.strings.reduce_join(self.int_to_char(integers))

encoder = CharacterEncoder(unique_chars)
max_text_len = max([len(text) for text in text_labels])

# Save results
results = {
    'dataset_info': {
        'total_samples': len(image_files),
        'unique_characters': len(unique_chars),
        'character_set': unique_chars,
        'max_sequence_length': max([len(text) for text in text_labels]),
        'min_sequence_length': min([len(text) for text in text_labels])
    },
    'config': {
        'batch_size': config.batch_size,
        'image_width': config.image_width,
        'image_height': config.image_height,
        'train_split': config.train_split
    },
    'tensorflow_version': tf.__version__,
    'keras_version': keras.__version__,
    'gpu_available': gpu_enabled,
    'gpu_devices': len(tf.config.list_physical_devices('GPU')),
    'gpu_memory_info': gpu_config.get_gpu_memory_info()
}

# Save to JSON
with open('text_recognition_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to text_recognition_results.json")
print(json.dumps(results, indent=2))