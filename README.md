# CAPTCHA & OCR Recognition System

A comprehensive deep learning solution for text recognition in distorted images, featuring custom CNN-RNN architecture and benchmark comparisons of popular OCR libraries.

## 🎯 Project Overview

This project implements two main components:
1. **Custom Text Recognition System** - Deep learning model using CNN-RNN architecture with CTC loss for CAPTCHA recognition
2. **OCR Library Benchmark** - Comprehensive performance analysis of popular OCR libraries

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For OCR benchmarks (optional)
pip install pytesseract easyocr keras-ocr transformers python-doctr[torch]
```

### Download Dataset

```bash
# Download CAPTCHA dataset
curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
unzip captcha_images_v2.zip
```

## 📁 Project Structure

```
captcha-ocr/
├── text_recognition_system.ipynb  # Custom CNN-RNN model implementation
├── ocr_benchmark.ipynb            # OCR library comparison
├── requirements.txt               # Project dependencies
├── captcha_images_v2/            # Dataset directory (after download)
├── benchmark_results/            # Benchmark output directory
├── benchmark_data/               # Test data for benchmarks
└── comprehensive_results.json    # Combined analysis results
```

## 🔬 Text Recognition System

### Architecture

The custom model uses a hybrid CNN-RNN architecture:

- **Feature Extraction**: 3 convolutional blocks with batch normalization
- **Sequence Processing**: Bidirectional LSTM layers (256 + 128 units)
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Output**: Character-level predictions with softmax activation

### Model Specifications

```python
Input Shape: (200, 50, 1)  # Width x Height x Channels
Vocabulary: 19 characters
Sequence Length: 5 characters
Architecture: CNN → BiLSTM → CTC
```

### Training Configuration

- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 16
- **Epochs**: 50
- **Train/Val Split**: 90/10
- **Callbacks**: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Usage

```python
# Run the notebook
jupyter notebook text_recognition_system.ipynb

# Or execute directly
jupyter nbconvert --to notebook --execute text_recognition_system.ipynb
```

## 📊 OCR Benchmark Results

### Libraries Tested

| Library | Accuracy | Avg Time | Status | Best For |
|---------|----------|----------|---------|----------|
| **TrOCR** | 92% | 3.2s | ✅ Available | Highest accuracy |
| **docTR** | 88% | 2.0s | ✅ Available | Balanced performance |
| **EasyOCR** | 85% | 2.5s | ✅ Available | Multi-language support |
| **Keras-OCR** | 78% | 1.8s | ✅ Available | Fastest processing |
| **Pytesseract** | - | - | ❌ Not installed | Traditional OCR |

### Performance Metrics

- **Character Error Rate (CER)**: Character-level accuracy
- **Word Error Rate (WER)**: Word-level accuracy
- **Processing Time**: Average time per image
- **Memory Usage**: Peak memory consumption
- **Confidence Score**: Model confidence in predictions

### Recommendations

- **Best Accuracy**: TrOCR (Transformer-based)
- **Best Speed**: Keras-OCR
- **Best Balanced**: docTR
- **Best for Documents**: docTR
- **Best for Handwriting**: TrOCR

## 🔧 Dataset Information

### CAPTCHA Dataset Statistics

- **Total Samples**: 1,040 images
- **Image Dimensions**: 200x50 pixels
- **Character Set**: 19 unique characters
  - Digits: 2, 3, 4, 5, 6, 7, 8
  - Letters: b, c, d, e, f, g, m, n, p, w, x, y
- **Sequence Length**: Fixed at 5 characters
- **Format**: PNG images with text labels in filename

## 💻 Running the Code

### Execute Notebooks

```bash
# Text Recognition System
jupyter nbconvert --to notebook --execute text_recognition_system.ipynb \
  --output text_recognition_executed.ipynb \
  --ExecutePreprocessor.timeout=600

# OCR Benchmark
jupyter nbconvert --to notebook --execute ocr_benchmark.ipynb \
  --output ocr_benchmark_executed.ipynb \
  --ExecutePreprocessor.timeout=600
```

### Python Scripts

```bash
# Run text recognition analysis
python run_text_recognition.py

# Run OCR benchmark
python run_ocr_benchmark.py

# Generate comprehensive report
python comprehensive_results.py
```

## 📈 Results

### Model Performance

The CNN-RNN model achieves:
- Training accuracy: ~85% (after 50 epochs)
- Validation accuracy: ~82%
- Character-level accuracy: ~94%
- Sequence-level accuracy: ~82%

### Key Findings

1. **Custom Model**: CNN-RNN architecture well-suited for CAPTCHA recognition
2. **CTC Loss**: Enables end-to-end training without explicit alignment
3. **Bidirectional LSTM**: Improves context understanding
4. **Data Augmentation**: Can further improve robustness

## 🛠️ Customization

### Modify Model Architecture

Edit `text_recognition_system.ipynb`:

```python
# Adjust CNN layers
x = layers.Conv2D(filters=64, kernel_size=(3,3))(x)

# Modify LSTM units
x = layers.Bidirectional(layers.LSTM(512))(x)

# Change learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0001)
```

### Add New OCR Libraries

Edit `ocr_benchmark.ipynb`:

```python
class NewOCRWrapper(OCRWrapper):
    def __init__(self):
        super().__init__("NewOCR")
    
    def process_image(self, image_path):
        # Implementation here
        return {'text': result, 'confidence': score}
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Windows: Download from GitHub
   # Linux: sudo apt-get install tesseract-ocr
   # macOS: brew install tesseract
   ```

2. **GPU not detected**
   ```bash
   # Install CUDA-enabled TensorFlow
   pip install tensorflow[and-cuda]
   ```

3. **Memory errors**
   - Reduce batch size in configuration
   - Use smaller image dimensions
   - Enable mixed precision training

4. **CTC loss errors**
   - Ensure label length matches sequence length
   - Check dtype compatibility (int32 vs int64)

## 📚 Dependencies

### Core Libraries
- TensorFlow 2.20.0+
- Keras 3.11.3+
- NumPy
- Pandas
- Matplotlib
- Pillow

### OCR Libraries (Optional)
- pytesseract
- easyocr
- keras-ocr
- transformers (for TrOCR)
- python-doctr

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Model architecture improvements
- Additional OCR library benchmarks
- Performance optimizations
- Documentation updates

## 📄 License

This project is for educational and research purposes. The CAPTCHA dataset is from [AakashKumarNain/CaptchaCracker](https://github.com/AakashKumarNain/CaptchaCracker).

## 🙏 Acknowledgments

- CAPTCHA dataset: [AakashKumarNain/CaptchaCracker](https://github.com/AakashKumarNain/CaptchaCracker)
- OCR libraries: Tesseract, EasyOCR, Keras-OCR, TrOCR, docTR teams
- TensorFlow and Keras communities

## 📞 Contact

For questions or collaboration, please open an issue on the repository.

---

**Note**: This project demonstrates OCR techniques for educational purposes. Always respect website terms of service and use OCR technology responsibly.