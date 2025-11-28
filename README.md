# 🧠 NeuroScan AI - Alzheimer's Detection System

<div align="center">

![NeuroScan AI Banner](https://via.placeholder.com/1600x400/1e293b/3b82f6?text=NeuroScan+AI+-+Alzheimer's+Detection+System)
*Professional 16:4 project thumbnail*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-2.0-green)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-99.2%25-brightgreen)]()

**Advanced Deep Learning System for Alzheimer's Disease Classification from MRI Scans**

</div>

## 📋 Table of Contents

- [🌟 Overview](#-overview)
- [🚀 Features](#-features)
- [📊 Performance Metrics](#-performance-metrics)
- [🛠️ Technical Architecture](#️-technical-architecture)
- [💻 Installation](#-installation)
- [🎯 Usage](#-usage)
- [📁 Project Structure](#-project-structure)
- [🔬 Model Details](#-model-details)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👨‍💻 Developer](#-developer)

## 🌟 Overview

NeuroScan AI is a state-of-the-art deep learning system designed for early detection and classification of Alzheimer's disease from MRI scans. Leveraging advanced convolutional neural networks and transfer learning techniques, this system achieves exceptional accuracy in classifying Alzheimer's disease into four distinct stages.

### 🎯 Key Highlights

- **🏆 99.2% Test Accuracy** on diverse MRI datasets
- **⚡ Real-time Analysis** with results in under 2 seconds
- **🔬 Four-Stage Classification** for comprehensive diagnosis
- **💻 Web-based Interface** for easy accessibility
- **🎓 Educational Focus** with detailed documentation

## 🚀 Features

### 🧩 Core Capabilities
- **🧠 MRI Image Analysis** - Automated processing of brain MRI scans
- **📊 Multi-class Classification** - Four Alzheimer's stages:
  - 🟢 No Impairment
  - 🟡 Very Mild Impairment  
  - 🟠 Mild Impairment
  - 🔴 Moderate Impairment
- **📈 Confidence Scoring** - Detailed probability distributions
- **🖼️ Image Preprocessing** - Automatic normalization and enhancement

### 💡 Advanced Features
- **🎨 Interactive Web Interface** - User-friendly dashboard
- **📱 Responsive Design** - Works on all devices
- **🔍 Real-time Processing** - Instant analysis and results
- **📋 Medical Recommendations** - AI-generated next steps
- **📊 Visualization Tools** - Charts and graphs for better understanding

## 📊 Performance Metrics

### 🏅 Model Performance
| Metric | Value | Description |
|--------|-------|-------------|
| **Test Accuracy** | 🎯 **99.18%** | Overall classification accuracy |
| **Validation Accuracy** | 🎯 **99.22%** | Validation set performance |
| **Precision** | 📊 **99.75%** | Average across all classes |
| **Recall** | 📊 **99.25%** | Average across all classes |
| **F1-Score** | 📊 **99.50%** | Harmonic mean of precision & recall |
| **AUC Score** | ⭐ **1.00** | Perfect classification capability |

### 📈 Classification Report
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Mild Impairment | 1.00 | 0.99 | 1.00 | 179 |
| Moderate Impairment | 1.00 | 1.00 | 1.00 | 12 |
| No Impairment | 0.99 | 1.00 | 0.99 | 640 |
| Very Mild Impairment | 1.00 | 0.98 | 0.99 | 448 |

## 🛠️ Technical Architecture

### 🏗️ System Architecture
```
NeuroScan AI System
├── Frontend (Web Interface)
│   ├── HTML5 + CSS3 + JavaScript
│   ├ responsive Design
│   └── Interactive Charts
├── Backend (Flask API)
│   ├── RESTful Endpoints
│   ├── Image Processing
│   └── Model Serving
└── Machine Learning Core
    ├── EfficientNetB0 Base
    ├── Custom Classification Head
    └── Transfer Learning
```

### 🔧 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Machine Learning** | TensorFlow 2.x, Keras | Model development & training |
| **Backend Framework** | Flask, Python 3.8+ | Web API and server |
| **Frontend** | HTML5, CSS3, JavaScript | User interface |
| **Image Processing** | OpenCV, PIL | MRI preprocessing |
| **Data Handling** | NumPy, Pandas | Numerical computations |
| **Visualization** | Chart.js | Results display |

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- TensorFlow 2.x
- Flask 2.0+

### 🚀 Quick Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/neuroscan-ai.git
   cd neuroscan-ai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv neuroscan_env
   source neuroscan_env/bin/activate  # On Windows: neuroscan_env\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the System**
   ```
   Open your browser and navigate to: http://localhost:5000
   ```

### 📦 Dependencies

```txt
tensorflow==2.10.0
flask==2.3.0
opencv-python==4.7.0.72
numpy==1.24.0
pandas==1.5.0
pillow==9.5.0
```

## 🎯 Usage

### 1. 🖼️ Image Upload
- Navigate to the Detection page
- Upload MRI brain scan images (JPEG, PNG)
- Supported formats: Standard medical imaging formats

### 2. 🔍 Analysis Process
- Automatic image preprocessing
- Deep learning model inference
- Real-time classification
- Confidence score calculation

### 3. 📊 Results Interpretation
- Clear classification output
- Confidence levels
- Medical recommendations
- Visual analytics

### 🎮 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload MRI image for analysis |
| `POST` | `/api/realtime_predict` | Real-time drawing analysis |
| `GET` | `/api/model/info` | Get model metadata |
| `GET` | `/api/health` | System health check |

## 📁 Project Structure

```
neuroscan-ai/
├── app.py                          # Main Flask application
├── model/                          # Machine learning models
│   ├── alzheimer_model.h5         # Trained model weights
│   └── model_architecture.py      # Model definition
├── static/                         # Static assets
│   ├── css/
│   ├── js/
│   └── images/
├── templates/                      # HTML templates
│   ├── index.html                 # Homepage
│   ├── detection.html             # Image upload interface
│   ├── results.html               # Analysis results
│   └── developer.html             # Documentation
├── utils/                          # Utility functions
│   ├── image_processing.py        # Image preprocessing
│   └── model_utils.py             # Model helper functions
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔬 Model Details

### 🧠 Architecture Overview

**Base Model**: EfficientNetB0 (Transfer Learning)
- **Pre-trained Weights**: ImageNet
- **Fine-tuning**: Layers from block5a_expand_activation onward
- **Input Shape**: 224×224×3

**Custom Classification Head**:
- Global Average Pooling
- Dense Layer (128 units) + BatchNorm + Dropout (0.5)
- Dense Layer (128 units) + BatchNorm + Dropout (0.5)
- Output Layer (4 units, Softmax)

### 🎯 Training Strategy

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Categorical Crossentropy
- **Callbacks**: Early Stopping, ReduceLROnPlateau
- **Regularization**: L2 (0.001), Dropout (0.5)
- **Data Augmentation**: Rotation, Flips, Brightness

### 📊 Dataset

- **Training Images**: 10,240 MRI scans
- **Test Images**: 1,279 MRI scans
- **Classes**: 4 Alzheimer's stages
- **Source**: Curated medical dataset

## 📈 Results

### 🎯 Confusion Matrix
```
Actual \ Predicted   Mild    Moderate   None    Very Mild
Mild                 178     0          1       0
Moderate             0       12         0       0  
None                 0       0          639     1
Very Mild            0       0          2       446
```

### 📊 Key Insights
- **Overall Accuracy**: 99.22% (1269/1279 correct)
- **Perfect Classification**: Moderate impairment class
- **Minor Errors**: 10 misclassifications out of 1279
- **Most Common Error**: Very Mild vs No impairment (2 cases)

### 🏆 AUC Scores
All classes achieved perfect AUC scores of 1.00, demonstrating excellent class separability and model performance.

## 🤝 Contributing

We welcome contributions to enhance NeuroScan AI! Here's how you can help:

### 🐛 Reporting Issues
- Use GitHub Issues to report bugs
- Include detailed descriptions and steps to reproduce

### 💡 Feature Requests
- Suggest new features or improvements
- Provide use cases and expected behavior

### 🔧 Development
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

### 📋 Contribution Guidelines
- Follow PEP 8 coding standards
- Write clear commit messages
- Update documentation as needed
- Add tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Medical Disclaimer

> **Important**: NeuroScan AI is designed for educational and research purposes only. It is not intended for clinical diagnosis or medical decision-making. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 👨‍💻 Developer

**Muhammad Hamza Nawaz**
- 🎓 **Data Scientist**
- 🏛️ **Al-khawarizmi Institute of Computer Science**
- 🏫 **University of Engineering and Technology (UET), Lahore**
- 📧 Email: iamhamzanawaz14@gmail.com
- 💼 LinkedIn:(https://www.linkedin.com/in/muhammad-hamza-nawaz-a434501b3/) 
- 🔗 GitHub: (https://github.com/hamzanawazsangha/NeuroScan-AI---Alzheimer-s-Detection-System)

### 🎓 Academic Affiliation
This project was developed as part of my academic journey at the prestigious **Al-khawarizmi Institute of Computer Science, University of Engineering and Technology (UET), Lahore**. The institute's emphasis on cutting-edge research and practical implementation provided the perfect environment for developing this advanced AI healthcare solution.

---

<div align="center">

### 🌟 **"Advancing Healthcare through Artificial Intelligence"** 🌟

**NeuroScan AI - Making Alzheimer's detection accessible and accurate**

[![UET Lahore](https://via.placeholder.com/100x100/8B0000/FFFFFF?text=UET)](https://uet.edu.pk)
[![KICS](https://via.placeholder.com/100x100/00008B/FFFFFF?text=KICS)](https://kics.edu.pk)

*Developed with ❤️ at Al-khawarizmi Institute of Computer Science, UET Lahore*

</div>

---

### 📞 Contact & Support

For questions, support, or collaboration opportunities:
- 📧 **Email**: iamhamzanawaz14@gmail.com
- 💬 **Issues**: [GitHub Issues](https://github.com/hamzanawazsangha/NeuroScan-AI---Alzheimer-s-Detection-System/issues) 
- 📚 **Documentation**: [Full Documentation](docs/)

### 🙏 Acknowledgments

- **Al-khawarizmi Institute of Computer Science** for academic support
- **UET Lahore** for research facilities
- **TensorFlow team** for excellent deep learning framework
- **Medical researchers** who contributed to the dataset

---

<div align="center">

**⭐ Don't forget to star this repository if you find it helpful!**

[![Star History Chart](https://api.star-history.com/svg?repos=hamzanawazsangha/neuroscan-ai&type=Date)](https://star-history.com/#hamzanawazsangha/neuroscan-ai&Date)

</div>
