# Movie Review Sentiment Analysis

## 🎬 Project Overview

This project focuses on classifying movie reviews as either positive or negative using various machine learning techniques. We employ a Bag of Words approach for text preprocessing and explore different classification algorithms to determine the sentiment of IMDB movie reviews.

## 📊 Dataset

We use the IMDB Dataset of 50K Movie Reviews, which consists of two columns:
- `review`: The text of the movie review
- `sentiment`: The sentiment of the review (positive or negative)

*Dataset Source: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)*

## 🛠️ Technologies Used

- Python
- pandas
- numpy
- scikit-learn

## 📌 Key Features

1. Data preprocessing using Bag of Words (CountVectorizer)
2. Implementation of multiple classification algorithms:
   - Random Forest
   - K-Nearest Neighbors (KNN)
   - Multinomial Naive Bayes
3. Performance evaluation using classification reports and confusion matrices

## 🚀 Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```
3. Run the Jupyter notebook or Python script

## 📊 Model Performance

We evaluated three different models:

### 1. Random Forest

- Achieved ~84% precision, recall, and F1-score for both positive and negative sentiments

![download](https://github.com/user-attachments/assets/8cd99139-4804-4a86-9ff3-65ec864b4d7e)

**How it works:** Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of individual trees. It uses:
- Bootstrap aggregating (bagging) to sample data points with replacement
- Random feature selection to decrease correlation between trees
- Voting mechanism to make final predictions

### 2. K-Nearest Neighbors (KNN)

- Performed less effectively with ~65% precision, recall, and F1-score

![image](https://github.com/user-attachments/assets/f27e4254-1429-412b-a8b7-06395c8e1e0b)

**How it works:** KNN is a non-parametric method used for classification and regression. For classification:
- It finds the K nearest neighbors to a given data point based on a distance metric (e.g., Euclidean distance)
- The majority class among these K neighbors determines the class of the data point
- In this project, we used K=10 and Euclidean distance

### 3. Multinomial Naive Bayes

- Matched the performance of Random Forest with ~84% precision, recall, and F1-score

![download](https://github.com/user-attachments/assets/69cc356c-77d5-47d9-8bd9-61d794e22a0e)

**How it works:** Multinomial Naive Bayes is a probabilistic learning method particularly suited for text classification. It:
- Assumes features (words in our case) are generated from a simple multinomial distribution
- Uses Bayes' theorem to predict the most likely class
- Calculates the probability of each class given the input features, assuming feature independence

## 🔍 Observations

- Random Forest and Multinomial Naive Bayes outperformed KNN significantly
- KNN's performance degraded due to the high-dimensional nature of text data
- Multinomial Naive Bayes is particularly well-suited for text classification tasks
- Random Forest's bootstrapping and feature importance contribute to its strong performance

## 📝 Text Preprocessing

### Bag of Words (BoW)

Bag of Words is a text representation method that describes the occurrence of words within a document. It involves two things:
1. A vocabulary of known words
2. A measure of the presence of known words

Key characteristics:
- It's called a "bag" of words because any information about the order or structure of words in the document is discarded
- The model is only concerned with whether known words occur in the document, not where in the document

### CountVectorizer

CountVectorizer is a scikit-learn class that implements the Bag of Words concept. It works as follows:

1. **Tokenization**: Splits the text into words (tokens)
2. **Vocabulary Building**: Creates a vocabulary of unique words across all documents
3. **Encoding**: Transforms each document into a vector where each position represents a word in the vocabulary, and the value represents the word count in the document

Example:
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
```

The resulting matrix X will be a sparse matrix where each row represents a document, each column represents a word in the vocabulary, and each cell contains the count of the word in the document.

## 🎓 Learning Outcomes

This project demonstrates:
1. The importance of choosing appropriate algorithms for text classification tasks
2. The strengths and weaknesses of different approaches when dealing with high-dimensional data
3. The effectiveness of Bag of Words and CountVectorizer for text preprocessing
4. How to implement and compare multiple machine learning models for a classification task

## 📚 Further Reading

- [K-Nearest Neighbors Algorithm in Python and Scikit-Learn](https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/)
- [Naive Bayes: Why is it Favoured for Text-Related Tasks?](https://analyticsindiamag.com/naive-bayes-why-is-it-favoured-for-text-related-tasks/)
- [Scikit-learn CountVectorizer Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [An Introduction to Bag of Words (BoW)](https://machinelearningmastery.com/gentle-introduction-bag-words-model/)
