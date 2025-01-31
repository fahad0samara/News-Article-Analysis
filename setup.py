from setuptools import setup, find_packages

setup(
    name="news-classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.2',
        'nltk>=3.6.3',
        'streamlit>=1.0.0',
        'joblib>=1.0.1',
        'langdetect>=1.0.9',
        'jupyter>=1.0.0',
        'notebook>=6.4.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning-based news article classifier",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/news-classifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
