# News Article Classifier

A machine learning-based news article classifier that categorizes articles into different categories (Technology, Business, Sports, Entertainment, Politics) using advanced NLP techniques.

## Features

- **Multi-Language Support**: Handles both English and French articles
- **Advanced Classification**: Uses ensemble of models (Gradient Boosting, Logistic Regression, SVM)
- **Context Analysis**: Extracts and analyzes contextual features
- **Key Phrase Extraction**: Identifies important phrases and topics
- **Interactive Interface**: User-friendly Streamlit web interface
- **Detailed Analysis**: Shows confidence scores and context indicators

## Project Structure

```
news-classifier/
├── models/                  # Trained model files
│   ├── gradient_boosting_model.joblib
│   └── ensemble_model.joblib
├── notebooks/              # Jupyter notebooks
│   ├── news_classifier.ipynb
│   └── bbc_news_classifier.ipynb
├── src/                    # Source code
│   ├── data/              # Data processing
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── features/          # Feature engineering
│   │   ├── __init__.py
│   │   └── context_features.py
│   ├── models/           # Model training
│   │   ├── __init__.py
│   │   └── train.py
│   └── interface/        # Web interface
│       ├── __init__.py
│       └── app.py
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup
└── README.md            # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/news-classifier.git
cd news-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Web Interface

```bash
streamlit run src/interface/app.py
```

### Using the Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open either:
- `notebooks/news_classifier.ipynb` for general usage
- `notebooks/bbc_news_classifier.ipynb` for BBC dataset specific analysis

## Models

The system uses an ensemble of models:
1. **Gradient Boosting Classifier**: Main classifier with context features
2. **Logistic Regression**: For probabilistic classification
3. **Support Vector Machine**: For additional classification strength

## Features

### Context Analysis
- Technology indicators
- Business terminology
- Sports-related terms
- Entertainment features
- Political context

### Key Phrase Extraction
- Bigram analysis
- Frequency-based extraction
- Custom stopword filtering
- Multi-language support

## Example Usage

```python
from src.interface.app import classify_article

# Example article
article = """
Apple reports record quarterly earnings as iPhone sales surge in emerging markets. 
The tech giant saw a 15% increase in revenue, largely driven by strong performance 
in India and Southeast Asia. CEO Tim Cook announced plans for expanding their AI initiatives.
"""

# Get classification
result = classify_article(article)
print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- BBC News Dataset for training data
- Scikit-learn for machine learning tools
- NLTK for natural language processing
- Streamlit for the web interface
