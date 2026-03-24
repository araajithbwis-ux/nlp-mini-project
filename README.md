# NLP Mini Project – Sentiment Analysis on IMDB Reviews

## Overview
This project implements a classical NLP pipeline to classify movie reviews as positive or negative.  
It compares the effect of **stopword removal** on classification accuracy.

### Pipeline
- TF‑IDF vectorization (max 5000 features)
- Logistic Regression classifier
- 80/20 train/test split

### Results
- Stopwords removed: **89.15%** accuracy
- Stopwords kept: **89.51%** accuracy

## Dataset
IMDB movie reviews (50,000 labeled reviews) from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

## Requirements
- Python 3.x
- pandas
- scikit-learn
- matplotlib

## How to Run
1. Download the dataset as `IMDB Dataset.csv` and place it in the same folder as the script.
2. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib
