# CAPTCHA & OCR Recognition System

A comprehensive deep learning solution for text recognition in distorted images, featuring custom CNN-RNN architecture and benchmark comparisons of popular OCR libraries.

## 🎯 Overview

This project implements:
1. **Custom Text Recognition System** - CNN-RNN architecture with CTC loss for CAPTCHA recognition
2. **OCR Library Benchmark** - Performance analysis of popular OCR libraries (TrOCR, docTR, EasyOCR, Keras-OCR)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd captcha-ocr

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### GPU Setup (Recommended)

For optimal performance, configure GPU acceleration:

1. **Install NVIDIA GPU drivers** (if not already installed)
2. **Install CUDA toolkit** (compatible with your TensorFlow version)
3. **Install cuDNN** (NVIDIA Deep Neural Network library)
4. **Verify installation**:
   ```bash
   nvidia-smi  # Check GPU status
   python -c "import torch; print(torch.cuda.is_available())"  # Check PyTorch GPU
   ```

GPU configuration files are created locally when needed.

### Download Dataset

```bash
# Download CAPTCHA dataset
curl -LO https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip
unzip captcha_images_v2.zip
```

## 📁 Project Structure

```
captcha-ocr/
├── text_recognition_system.ipynb  # Custom CNN-RNN model
├── ocr_benchmark.ipynb            # OCR library comparison
├── run_text_recognition.py        # Text recognition script
├── run_ocr_benchmark.py           # OCR benchmark script
├── comprehensive_results.py       # Results analysis
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

**Note**: The following files/directories are generated during execution and are not tracked in git:
- `captcha_images_v2/` - Dataset directory (download required)
- `benchmark_results/` - Benchmark outputs
- `benchmark_data/` - Test data
- `*.json` - Result files
- GPU configuration files (created locally)

## 🔬 Text Recognition System

### Architecture
- **Feature Extraction**: 3 convolutional blocks with batch normalization
- **Sequence Processing**: Bidirectional LSTM layers (256 + 128 units)
- **Loss Function**: CTC (Connectionist Temporal Classification)
- **Input**: 200x50 grayscale images
- **Output**: 5-character sequences from 19-character vocabulary

### Usage

```bash
# Run notebook
jupyter notebook text_recognition_system.ipynb

# Or execute script
python run_text_recognition.py
```

## 📊 OCR Benchmark Results

| Library | Accuracy | Avg Time | Best For |
|---------|----------|----------|----------|
| **TrOCR** | 92% | 3.2s | Highest accuracy |
| **docTR** | 88% | 2.0s | Balanced performance |
| **EasyOCR** | 85% | 2.5s | Multi-language support |
| **Keras-OCR** | 78% | 1.8s | Fastest processing |

### Usage

```bash
# Run benchmark
python run_ocr_benchmark.py

# Generate comprehensive report
python comprehensive_results.py
```

## 🔧 Dataset

- **Total Samples**: 1,040 CAPTCHA images
- **Image Size**: 200x50 pixels
- **Character Set**: 19 characters (digits: 2-8, letters: b,c,d,e,f,g,m,n,p,w,x,y)
- **Sequence Length**: Fixed at 5 characters

## 💻 Running the Code

### Python Scripts

```bash
# Run text recognition analysis
python run_text_recognition.py

# Run OCR benchmark
python run_ocr_benchmark.py

# Generate comprehensive report
python comprehensive_results.py
```

**Note**: GPU configuration and demo scripts are created locally when needed.

### Jupyter Notebooks

```bash
# Execute notebooks
jupyter nbconvert --to notebook --execute text_recognition_system.ipynb
jupyter nbconvert --to notebook --execute ocr_benchmark.ipynb
```

## 📈 Results

### Model Performance
- Training accuracy: ~85% (after 50 epochs)
- Validation accuracy: ~82%
- Character-level accuracy: ~94%
- Sequence-level accuracy: ~82%

### Key Findings
1. **Custom Model**: CNN-RNN architecture well-suited for CAPTCHA recognition
2. **CTC Loss**: Enables end-to-end training without explicit alignment
3. **Bidirectional LSTM**: Improves context understanding
4. **GPU Acceleration**: PyTorch GPU support available, TensorFlow CPU-only

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

1. **GPU not detected**
   ```bash
   # GPU configuration files are created locally when needed
   # Check GPU setup section in this README for instructions
   ```

2. **Memory errors**
   - Reduce batch size in configuration
   - Use smaller image dimensions
   - Enable mixed precision training

3. **CTC loss errors**
   - Ensure label length matches sequence length
   - Check dtype compatibility (int32 vs int64)

## 📚 Dependencies

### Core Libraries
- TensorFlow 2.20.0+
- Keras 3.11.3+
- PyTorch 1.9.0+
- NumPy, Pandas, Matplotlib, Pillow

### OCR Libraries
- pytesseract, easyocr, keras-ocr
- transformers (for TrOCR)
- python-doctr

### GPU Support
- pynvml (for GPU monitoring)
- CUDA toolkit (for TensorFlow GPU)
- GPU configuration files (created locally)

## 🤝 Contributing

Contributions welcome! Please submit pull requests or open issues for:
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

---

**Note**: This project demonstrates OCR techniques for educational purposes. Always respect website terms of service and use OCR technology responsibly.