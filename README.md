# News Article Classifier 📰 

An advanced machine learning system that automatically classifies news articles into categories using state-of-the-art NLP techniques and ensemble learning.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange)
![NLTK](https://img.shields.io/badge/NLTK-3.6%2B-green)

## 🌟 Key Features

- **🔍 Multi-Language Support**
  - English and French article classification
  - Automatic language detection
  - Language-specific preprocessing

- **🤖 Advanced Classification**
  - Ensemble of models:
    - Gradient Boosting (main classifier)
    - Logistic Regression (probability calibration)
    - Support Vector Machine (boundary refinement)
  - Context-aware feature extraction
  - High accuracy across categories

- **📊 Detailed Analysis**
  - Category confidence scores
  - Key phrase extraction
  - Context indicators
  - Topic modeling

- **💻 User Interface**
  - Clean, modern Streamlit interface
  - Real-time classification
  - Detailed result visualization
  - Batch processing support

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/fahad0samara/News-Article-Analysis.git
cd News-Article-Analysis
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### 🎯 Usage

#### Web Interface

1. Start the Streamlit app:
```bash
streamlit run src/interface/app.py
```

2. Open your browser and go to `http://localhost:8501`

3. Enter or paste your article text

4. Get instant classification results!

#### Python API

```python
from src.models.classifier import NewsClassifier

# Initialize classifier
classifier = NewsClassifier()

# Classify an article
article = """
Apple reports record quarterly earnings as iPhone sales surge in emerging markets. 
The tech giant saw a 15% increase in revenue, largely driven by strong performance 
in India and Southeast Asia.
"""

result = classifier.classify(article)
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Key Phrases: {', '.join(result['key_phrases'])}")
```

## 📊 Model Performance

Our ensemble model achieves:
- **93%** accuracy on tech articles
- **91%** accuracy on business articles
- **95%** accuracy on sports articles
- **89%** accuracy on entertainment articles
- **90%** accuracy on politics articles

## 🔍 Example Classifications

1. **Technology Article**
```
Microsoft unveils groundbreaking AI features for Windows 12, integrating 
advanced machine learning capabilities across the operating system.
```
- Category: Technology (96% confidence)
- Key phrases: AI features, machine learning, Windows 12

2. **Business Article**
```
Goldman Sachs reports Q4 earnings beating market expectations, with 
revenue up 25% year-over-year.
```
- Category: Business (94% confidence)
- Key phrases: earnings, revenue, market expectations

## 🛠️ Project Structure

```
news-classifier/
├── models/                  # Trained model files
├── notebooks/              # Jupyter notebooks
├── src/
│   ├── data/              # Data processing
│   ├── features/          # Feature engineering
│   ├── models/           # Model training
│   └── interface/        # Web interface
├── requirements.txt
├── setup.py
└── README.md
```

## 📈 Advanced Features

### Context Analysis
- Company name recognition
- Technical terminology detection
- Financial metric analysis
- Sports-specific term identification
- Entertainment industry knowledge

### Key Phrase Extraction
- TF-IDF based extraction
- Noun phrase analysis
- Named entity recognition
- Bigram/trigram analysis

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch:
```bash
git checkout -b feature/amazing-feature
```
3. Commit your changes:
```bash
git commit -m 'Add amazing feature'
```
4. Push to the branch:
```bash
git push origin feature/amazing-feature
```
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- BBC News Dataset for training data
- Scikit-learn team for machine learning tools
- NLTK team for NLP capabilities
- Streamlit team for the amazing web framework

## 📧 Contact

Fahad - fahad.samara@gmail.com

Project Link: [https://github.com/fahad0samara/News-Article-Analysis](https://github.com/fahad0samara/News-Article-Analysis)

## 🔮 Future Enhancements

- [ ] Add support for more languages
- [ ] Implement deep learning models
- [ ] Add API endpoint
- [ ] Create Docker container
- [ ] Add real-time news feed analysis
