# Movie Review Sentiment Classification using NLP and Machine Learning

A Python data science project that performs sentiment classification on IMDb movie reviews using Natural Language Processing and Machine Learning. The model predicts whether a movie review is **positive** or **negative** based on the review text.

This project also includes feature engineering, exploratory data analysis, supervised learning, unsupervised learning using K-Means clustering, dimensionality reduction using TruncatedSVD, and model persistence using Joblib.

---

## Repository Link

https://github.com/GSUS2K/movie-review-sentiment-analysis

---

## Project Objectives

- To load and analyze the IMDb movie review dataset.
- To clean and preprocess text data using NLP techniques.
- To create additional text-based features from the review column.
- To perform exploratory data analysis using Matplotlib and Seaborn.
- To generate word clouds for positive and negative reviews.
- To convert review text into numerical features using TF-IDF.
- To train and compare supervised machine learning models.
- To apply K-Means clustering for unsupervised learning.
- To visualize review clusters using TruncatedSVD.
- To save the final trained model using Joblib.

---

## Dataset

The dataset used is the **IMDb Dataset of 50K Movie Reviews**.

The original dataset contains two columns:

| Column | Description |
|---|---|
| `review` | Text content of the movie review |
| `sentiment` | Sentiment label: positive or negative |

Although the original dataset has only two columns, the `review` text is transformed into thousands of numerical TF-IDF features for machine learning. Additional engineered features were also created to improve analysis.

---

## Engineered Features

The following additional columns were created from the review text:

| Feature | Description |
|---|---|
| `character_count` | Total number of characters in the review |
| `word_count` | Total number of words in the review |
| `sentence_count` | Approximate number of sentences in the review |
| `exclamation_count` | Number of exclamation marks in the review |
| `question_count` | Number of question marks in the review |
| `average_word_length` | Average length of words in the review |
| `clean_review` | Cleaned and preprocessed version of the review |
| `clean_review_length` | Number of words after text cleaning |
| `cluster` | Cluster assigned using K-Means clustering |

---

## Technologies and Libraries Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK
- WordCloud
- Joblib

---

## Machine Learning Models Used

### Supervised Learning

- Naive Bayes
- Logistic Regression
- Linear Support Vector Machine
- K-Nearest Neighbors

### Unsupervised Learning

- K-Means Clustering

### Dimensionality Reduction

- TruncatedSVD

---

## Evaluation Metrics

The classification models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

## Project Workflow

1. Import required Python libraries.
2. Load the IMDb movie review dataset.
3. Check dataset shape, missing values, and duplicate records.
4. Remove duplicate records.
5. Convert sentiment labels into numerical values.
6. Create additional engineered features.
7. Clean and preprocess review text.
8. Perform exploratory data analysis.
9. Generate positive and negative word clouds.
10. Convert cleaned text into numerical features using TF-IDF.
11. Train supervised machine learning models.
12. Evaluate and compare model performance.
13. Build a Scikit-learn machine learning pipeline.
14. Test the model with custom movie reviews.
15. Save the trained model using Joblib.
16. Apply K-Means clustering.
17. Visualize clusters using TruncatedSVD.
18. Save enhanced dataset and output graphs.

---

## Text Preprocessing

The review text was cleaned using the following steps:

- Converted text to lowercase.
- Removed HTML tags.
- Removed URLs.
- Removed numbers and special characters.
- Removed stopwords.
- Removed common movie-related words.
- Applied lemmatization.
- Created a cleaned review column.

Lemmatization was used instead of stemming so that words remain readable and meaningful. For example, words like `excellent`, `beautiful`, and `terrible` remain understandable instead of being shortened into incomplete root forms.

---

## Exploratory Data Analysis

The project includes the following EDA tasks:

- Sentiment distribution analysis
- Review length distribution
- Average review length by sentiment
- Character count comparison
- Word count comparison
- Exclamation count comparison
- Correlation heatmap of engineered features
- Positive review word cloud
- Negative review word cloud
- Top positive sentiment words
- Top negative sentiment words

---

## Model Building

The cleaned review text was converted into numerical features using **TF-IDF Vectorization**. The transformed data was then used to train multiple machine learning models.

The models used were:

| Model | Purpose |
|---|---|
| Naive Bayes | Baseline model for text classification |
| Logistic Regression | Linear model for binary classification |
| Linear SVM | Suitable for high-dimensional text data |
| KNN | Distance-based classification model |

A Scikit-learn pipeline was also created to combine TF-IDF vectorization and Logistic Regression into a single reusable workflow.

---

## Project Folder Structure

```text
Movie_Sentiment_Project/
│
├── dataset/
│   └── IMDB Dataset.csv
│
├── notebook/
│   └── movie_sentiment_project.ipynb
│
├── models/
│   └── movie_sentiment_pipeline.pkl
│
├── outputs/
│   ├── enhanced_imdb_reviews.csv
│   ├── sentiment_distribution.png
│   ├── review_length_distribution.png
│   ├── average_review_length.png
│   ├── character_count_by_sentiment.png
│   ├── word_count_by_sentiment.png
│   ├── exclamation_count_by_sentiment.png
│   ├── correlation_heatmap.png
│   ├── positive_wordcloud.png
│   ├── negative_wordcloud.png
│   ├── naive_bayes_confusion_matrix.png
│   ├── logistic_regression_confusion_matrix.png
│   ├── model_accuracy_comparison.png
│   ├── cluster_visualization.png
│   ├── actual_sentiment_visualization.png
│   ├── top_positive_words.png
│   └── top_negative_words.png
│
├── report/
│   └── final_report.pdf
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/GSUS2K/movie-review-sentiment-analysis.git
```

### 2. Open the Project Folder

```bash
cd movie-review-sentiment-analysis
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Open Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open and run the notebook:

```text
notebook/movie_sentiment_project.ipynb
```

---

## Sample Prediction

Example positive review:

```python
predict_sentiment("The movie was amazing and the acting was excellent.")
```

Output:

```text
Positive Review
```

Example negative review:

```python
predict_sentiment("The movie was boring, slow and disappointing.")
```

Output:

```text
Negative Review
```

---

## Model Persistence

The final machine learning pipeline was saved using Joblib:

```python
joblib.dump(sentiment_pipeline, "../models/movie_sentiment_pipeline.pkl")
```

The saved model can be loaded later and used for predicting the sentiment of new movie reviews.

---

## Results

The project successfully demonstrates:

- Text preprocessing using NLP
- Feature engineering from review text
- Exploratory data analysis using visualizations
- Sentiment classification using supervised machine learning
- Model comparison using evaluation metrics
- Review clustering using K-Means
- Cluster visualization using TruncatedSVD
- Model saving using Joblib

The best-performing model can be selected based on the accuracy, precision, recall, and F1-score values obtained in the notebook.

---

## Conclusion

This project demonstrates a complete Python data science workflow for NLP-based sentiment classification. The original IMDb dataset contained only two columns, but additional features were generated using feature engineering. The review text was cleaned, transformed using TF-IDF, and classified using machine learning models.

Supervised models such as Naive Bayes, Logistic Regression, Linear SVM, and KNN were used for sentiment classification. K-Means clustering was also applied for unsupervised learning, and TruncatedSVD was used for two-dimensional cluster visualization.

This project covers important concepts such as data cleaning, data manipulation, exploratory data analysis, visualization, supervised learning, unsupervised learning, pipeline creation, model evaluation, and model persistence.

---

## Future Scope

- Use deep learning models such as LSTM or BERT.
- Add neutral sentiment classification.
- Deploy the model as a web application using Streamlit.
- Use recent reviews from OTT platforms or movie review websites.
- Improve clustering using advanced sentence embeddings.
- Add multilingual sentiment analysis.

---
