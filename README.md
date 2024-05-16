# DS5500_Capstone_Sentiment_Analysis

## Sentiment Analysis on Twitter Data: Exploring Abortion Discourse

Authors: Alan Cheung and Bezawit Ayalew

### Introduction

In this project, we delve into sentiment analysis on a Twitter dataset centered around discussions regarding abortion. Our goal is to utilize Python and various natural language processing (NLP) techniques to examine the sentiment expressed in tweets related to abortion. By analyzing and classifying the sentiment of these tweets, we aim to uncover insights into public opinion and attitudes surrounding this sensitive topic.

### Files Included

1. **Data Preprocessing_Classic ML Models_User Interface.ipynb**: This notebook contains code for data preprocessing and building classic machine learning models for sentiment analysis on the cleaned Twitter data. It also includes the development of a user interface (UI) for interacting with the a selected trained model.

2. **BERTopic_sentiment_analysis.ipynb**: This notebook focuses on sentiment analysis using the BERTopic approach. BERTopic is a topic modeling technique based on BERT embeddings. The notebook covers preprocessing, topic modeling using BERTopic, and sentiment analysis with a pre-trained DistilBERT model.

3. **LDA_DistilBERT_Models.ipynb**: This notebook explores sentiment analysis using a combination of Latent Dirichlet Allocation (LDA) and DistilBERT models. LDA is a topic modeling technique, while DistilBERT is a distilled version of the BERT model. The notebook includes preprocessing, topic modeling with LDA, training a DistilBERT model for topic modeling, and sentiment analysis with a pre-trained DistilBERT model.

4. **LDA_DistilBERT_functions.py**: This Python script contains reusable functions for preprocessing text data, performing LDA topic modeling, and conducting sentiment analysis using DistilBERT. It provides modularized code that can be imported and used in the notebooks or other scripts for analysis.

5. **data_preprocessing_cleaned.py**: This Python script contains resusable functions for cleaning, preprocessing, and visualizing text data. It provides modularized code that can be imported and used in the notebooks or other scripts for analysis.

6. **twitter_data.csv**: This dataset was sourced from Kaggle, encompassing a collection of 415,000 tweets spanning from July 1, 2022, to August 28, 2022, centered around the overturn of Roe v. Wade. Stored in a .csv format, the dataset is approximately 243 MB in size, structured into 20 columns across 415,373 rows. Each row represents an individual tweet, with columns detailing both user-specific information (such as follower count, user ID, location, etc.) and tweet-specific data (including hashtags, posting date, number of retweets, and more). This is the dataset used for this project. Source: https://www.kaggle.com/datasets/bwandowando/-roe-v-wade-twitter-dataset/data

### How to Use

1. **Clone the Repository**: Clone this repository to your local machine using:

    ```
    git clone https://github.com/aycheung/DS5500_Capstone_Sentiment_Analysis.git
    ```

2. **Navigate to the Directory**: Enter the project directory:

    ```
    cd DS5500_Capstone_Sentiment_Analysis
    ```

3. **Run the Notebooks**: Execute the notebooks in your preferred environment. These notebooks contain the code for loading, cleaning, and analyzing the Twitter data.

### Dependencies

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, matplotlib, seaborn, re, tkinter, nltk, torch, sklearn, transformers, gensim, pyLDAvis, bertopic
