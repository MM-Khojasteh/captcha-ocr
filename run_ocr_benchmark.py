import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

# Import GPU configuration
from gpu_config import gpu_config

# Configure GPU before importing other libraries
gpu_config.set_environment_variables()
gpu_config.optimize_for_inference()

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

print("OCR Benchmark - Simplified Version")
print("="*50)

# Configuration
class BenchmarkConfig:
    def __init__(self):
        self.output_dir = Path('./benchmark_results')
        self.data_dir = Path('./benchmark_data')
        self.test_texts = [
            "Hello World 123",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "abcdefghijklmnopqrstuvwxyz", 
            "0123456789",
            "The quick brown fox",
            "Email: user@example.com",
            "Price: $99.99"
        ]

config = BenchmarkConfig()
config.output_dir.mkdir(exist_ok=True)
config.data_dir.mkdir(exist_ok=True)

# Generate test images
def create_text_image(text, filename, size=(800, 100)):
    """Create a high-quality test image with text optimized for OCR"""
    # Create high-resolution image
    scale_factor = 3
    high_res_size = (size[0] * scale_factor, size[1] * scale_factor)
    img = Image.new('RGB', high_res_size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Try multiple fonts for better compatibility
    font_size = 24 * scale_factor
    fonts_to_try = [
        "arial.ttf",
        "calibri.ttf", 
        "times.ttf",
        "verdana.ttf"
    ]
    
    font = None
    for font_name in fonts_to_try:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()
    
    # Center the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (high_res_size[0] - text_width) // 2
    y = (high_res_size[1] - text_height) // 2
    
    # Draw text with high contrast
    draw.text((x, y), text, fill='black', font=font)
    
    # Resize back to original size with high quality
    img = img.resize(size, Image.Resampling.LANCZOS)
    
    # Apply slight sharpening for better OCR
    from PIL import ImageFilter
    img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    
    img.save(filename, 'PNG', optimize=True)
    return filename

def enhance_image_for_ocr(image_path):
    """Enhance image for better OCR recognition"""
    try:
        import cv2
        import numpy as np
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return image_path
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (1, 1), 0)
        
        # Apply sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(blurred, -1, kernel)
        
        # Convert back to RGB
        enhanced_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        # Save enhanced image
        enhanced_path = image_path.replace('.png', '_enhanced.png')
        cv2.imwrite(enhanced_path, enhanced_rgb)
        
        return enhanced_path
    except Exception as e:
        print(f"  Enhancement failed: {e}")
        return image_path

print("\nGenerating test images...")
test_images = []
for i, text in enumerate(config.test_texts):
    filename = config.data_dir / f"test_{i}.png"
    create_text_image(text, filename)
    test_images.append({'path': str(filename), 'text': text})
    print(f"  Created: {filename.name}")

print(f"\nGenerated {len(test_images)} test images")

# Test with available OCR libraries
results = []

# Test Pytesseract
try:
    import pytesseract
    # Configure Tesseract path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    print("\nTesting Pytesseract...")
    
    for img_data in test_images:
        start_time = time.time()
        try:
            # Enhance image for better OCR
            enhanced_path = enhance_image_for_ocr(img_data['path'])
            img = Image.open(enhanced_path)
            
            # Use Tesseract with optimized settings
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@.:$!() '
            recognized_text = pytesseract.image_to_string(img, config=custom_config).strip()
            process_time = time.time() - start_time
            
            results.append({
                'library': 'Pytesseract',
                'image': Path(img_data['path']).name,
                'ground_truth': img_data['text'],
                'recognized': recognized_text,
                'match': recognized_text == img_data['text'],
                'time': process_time
            })
            print(f"  Processed: {Path(img_data['path']).name}")
        except Exception as e:
            print(f"  Error: {e}")
except ImportError:
    print("\nPytesseract not available - skipping")

# Test EasyOCR
try:
    import easyocr
    print("\nTesting EasyOCR...")
    # Use GPU configuration
    use_gpu = gpu_config.get_easyocr_gpu_config()
    print(f"  EasyOCR GPU: {'Enabled' if use_gpu else 'Disabled'}")
    reader = easyocr.Reader(['en'], gpu=use_gpu)
    
    for img_data in test_images:
        start_time = time.time()
        try:
            result = reader.readtext(img_data['path'])
            recognized_text = ' '.join([text for _, text, _ in result])
            process_time = time.time() - start_time
            
            results.append({
                'library': 'EasyOCR',
                'image': Path(img_data['path']).name,
                'ground_truth': img_data['text'],
                'recognized': recognized_text,
                'match': recognized_text == img_data['text'],
                'time': process_time
            })
            print(f"  Processed: {Path(img_data['path']).name}")
        except Exception as e:
            print(f"  Error: {e}")
except ImportError:
    print("\nEasyOCR not available - skipping")

# Test Keras-OCR (with NumPy 2.0 compatibility fix)
try:
    import keras_ocr
    import tensorflow as tf
    import numpy as np
    
    print("\nTesting Keras-OCR...")
    
    # Fix NumPy 2.0 compatibility issues
    if not hasattr(np, 'sctypes'):
        np.sctypes = {
            'int': [np.int8, np.int16, np.int32, np.int64],
            'uint': [np.uint8, np.uint16, np.uint32, np.uint64],
            'float': [np.float16, np.float32, np.float64],
            'complex': [np.complex64, np.complex128],
            'others': [bool, object, bytes, str]
        }
    
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # Create pipeline with error handling
    try:
        pipeline = keras_ocr.pipeline.Pipeline()
        print("  Keras-OCR pipeline loaded successfully")
    except Exception as pipeline_error:
        print(f"  Pipeline error: {pipeline_error}")
        # Try alternative approach
        pipeline = None
    
    if pipeline:
        for img_data in test_images:
            start_time = time.time()
            try:
                # Enhance image for better recognition
                enhanced_path = enhance_image_for_ocr(img_data['path'])
                image = keras_ocr.tools.read(enhanced_path)
                prediction_groups = pipeline.recognize([image])
                recognized_text = ' '.join([text for text, _ in prediction_groups[0]])
                process_time = time.time() - start_time
                
                results.append({
                    'library': 'Keras-OCR',
                    'image': Path(img_data['path']).name,
                    'ground_truth': img_data['text'],
                    'recognized': recognized_text,
                    'match': recognized_text == img_data['text'],
                    'time': process_time
                })
                print(f"  Processed: {Path(img_data['path']).name}")
            except Exception as e:
                print(f"  Error processing {Path(img_data['path']).name}: {e}")
    else:
        print("  Skipping Keras-OCR due to pipeline initialization error")
        
except ImportError:
    print("\nKeras-OCR not available - skipping")
except Exception as e:
    print(f"\nKeras-OCR error: {e}")

# Test TrOCR (using transformers with improved preprocessing)
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch
    import cv2
    import numpy as np
    
    print("\nTesting TrOCR...")
    # Load TrOCR model
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    
    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() and gpu_config.gpu_available else 'cpu')
    model.to(device)
    print(f"  TrOCR Device: {device}")
    
    def preprocess_image_for_trocr(image_path):
        """Preprocess image for better TrOCR recognition"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
            
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to RGB for TrOCR
        rgb_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(rgb_image)
    
    for img_data in test_images:
        start_time = time.time()
        try:
            # Try original image first
            image = Image.open(img_data['path']).convert('RGB')
            
            # Preprocess for better recognition
            processed_image = preprocess_image_for_trocr(img_data['path'])
            if processed_image is not None:
                image = processed_image
            
            # Resize image if too large (TrOCR works better with smaller images)
            if image.size[0] > 512 or image.size[1] > 512:
                image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
            generated_ids = model.generate(pixel_values, max_length=50, num_beams=5)
            recognized_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            process_time = time.time() - start_time
            
            results.append({
                'library': 'TrOCR',
                'image': Path(img_data['path']).name,
                'ground_truth': img_data['text'],
                'recognized': recognized_text,
                'match': recognized_text == img_data['text'],
                'time': process_time
            })
            print(f"  Processed: {Path(img_data['path']).name}")
        except Exception as e:
            print(f"  Error processing {Path(img_data['path']).name}: {e}")
except ImportError:
    print("\nTrOCR (transformers) not available - skipping")
except Exception as e:
    print(f"\nTrOCR error: {e}")

# Test docTR
try:
    import doctr
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
    
    print("\nTesting docTR...")
    # Load docTR model
    model = ocr_predictor(pretrained=True)
    
    for img_data in test_images:
        start_time = time.time()
        try:
            doc = DocumentFile.from_images(img_data['path'])
            result = model(doc)
            recognized_text = ' '.join([word.value for page in result.pages for block in page.blocks for line in block.lines for word in line.words])
            process_time = time.time() - start_time
            
            results.append({
                'library': 'docTR',
                'image': Path(img_data['path']).name,
                'ground_truth': img_data['text'],
                'recognized': recognized_text,
                'match': recognized_text == img_data['text'],
                'time': process_time
            })
            print(f"  Processed: {Path(img_data['path']).name}")
        except Exception as e:
            print(f"  Error: {e}")
except ImportError:
    print("\ndocTR not available - skipping")
except Exception as e:
    print(f"\ndocTR error: {e}")

# Save results
if results:
    df = pd.DataFrame(results)
    
    # Calculate statistics
    print("\n" + "="*70)
    print("COMPREHENSIVE OCR BENCHMARK RESULTS")
    print("="*70)
    
    # Create comparison table
    print(f"\n{'Library':<12} {'Accuracy':<10} {'Avg Time':<10} {'Best For':<25}")
    print("-" * 70)
    
    # Sort by accuracy (descending)
    sorted_libs = sorted(df['library'].unique(), 
                        key=lambda x: df[df['library'] == x]['match'].mean(), 
                        reverse=True)
    
    best_for = {
        'TrOCR': 'Highest accuracy',
        'docTR': 'Balanced performance', 
        'EasyOCR': 'Multi-language support',
        'Keras-OCR': 'Fastest processing',
        'Pytesseract': 'Traditional OCR'
    }
    
    for lib in sorted_libs:
        lib_data = df[df['library'] == lib]
        accuracy = lib_data['match'].mean() * 100
        avg_time = lib_data['time'].mean()
        print(f"{lib:<12} {accuracy:>6.1f}%   {avg_time:>6.3f}s   {best_for.get(lib, 'General purpose'):<25}")
    
    print("\n" + "="*70)
    print("DETAILED RESULTS")
    print("="*70)
    
    for lib in df['library'].unique():
        lib_data = df[df['library'] == lib]
        accuracy = lib_data['match'].mean() * 100
        avg_time = lib_data['time'].mean()
        
        print(f"\n{lib}:")
        print(f"  Accuracy: {accuracy:.1f}%")
        print(f"  Avg Time: {avg_time:.3f}s")
        print(f"  Correct: {lib_data['match'].sum()}/{len(lib_data)}")
    
    # Save to CSV
    csv_path = config.output_dir / 'benchmark_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save to JSON
    json_results = {
        'summary': {
            lib: {
                'accuracy': float(df[df['library'] == lib]['match'].mean()),
                'avg_time': float(df[df['library'] == lib]['time'].mean()),
                'total_tests': len(df[df['library'] == lib])
            }
            for lib in df['library'].unique()
        },
        'details': df.to_dict('records')
    }
    
    json_path = config.output_dir / 'benchmark_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"JSON results saved to: {json_path}")
else:
    print("\nNo OCR libraries available for testing")
    print("Install with: pip install pytesseract easyocr")

print("\nBenchmark complete!")