
"""
This file contains the code for preprocessing the data for LDA topic modeling,
running the LDA model, training a DistilBERT model, and running sentiment
analysis with DistilBERT
"""

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import pandas as pd


class DataPreprocessor:
    """
    Handles the cleaning and preprocessing of textual data.
    Initialization requires the path to a CSV file containing the data.
    """

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None

    def clean_data(self):
        """
        Loads and cleans the data using the TweetCleaner class.
        Applies lemmatization preprocessing to the 'cleaned_text' column.
        """
        import data_preprocessing_cleaned as clean
        df = clean.TweetCleaner(self.csv_file)
        cleaned_df = df.clean_tweets()
        cleaned_df['preprocessed_text'] = cleaned_df['cleaned_text'].apply(clean.preprocess_text_lemmatization)
        self.df = cleaned_df

    def preprocess_text(self):
        """
        Further processes the preprocessed text by removing stopwords,
        applying lemmatization, and tokenizing.
        """
        import nltk
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import RegexpTokenizer

        def preprocess(text):
            stop_words = set(stopwords.words('english'))
            additional_stopwords = {'u', 'would', 'amp', 'im', 'ur'}
            stop_words.update(additional_stopwords)
            tokenizer = RegexpTokenizer(r'\w+')
            lemmatizer = WordNetLemmatizer()
            tokens = tokenizer.tokenize(text.lower())
            return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and len(token) > 1]

        self.df['processed_text'] = self.df['cleaned_text'].apply(preprocess)

    def remove_retweets(self):
        '''
        Removes retweets from the dataframe

        '''
        self.df = self.df[self.df['is_retweet'] != True]


class LDATopicModeler:
    """
    Encapsulates the LDA topic modeling process including model training,
    coherence score calculation, and visualization of topics.
    """

    def __init__(self, df, documents:str):
        self.dataframe = df
        self.documents = df[documents]
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.coherence_values = []
        self.model_list = []

    def create_corpus_and_dictionary(self):
        """
        Creates a Gensim dictionary and corpus from the processed documents.
        """
        from gensim.corpora import Dictionary
        self.dictionary = Dictionary(self.documents)
        self.dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]

    # def find_optimal_number_of_topics(self, start=2, limit=40, step=3):
    #     """
    #     Computes coherence values for various number of topics to find the optimal number.
    #     """
    #     from gensim.models.coherencemodel import CoherenceModel
    #     from gensim.models.ldamodel import LdaModel

    #     coherence_values = []
    #     model_list = []
    #     for num_topics in range(start, limit, step):
    #         model = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary)
    #         model_list.append(model)
    #         coherencemodel = CoherenceModel(model=model, texts=self.documents, dictionary=self.dictionary, coherence='c_v')
    #         coherence_values.append(coherencemodel.get_coherence())
    #     return model_list, coherence_values

    def compute_coherence_values(self, limit=40, start=2, step=6):
        """
        Computes coherence values for various number of topics to find the optimal number.
        """
        from gensim.models.coherencemodel import CoherenceModel
        from gensim.models.ldamodel import LdaModel

        for num_topics in range(start, limit, step):
            model = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary)
            self.model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=self.documents, dictionary=self.dictionary, coherence='c_v')
            self.coherence_values.append(coherencemodel.get_coherence())

    def plot_optimal_number_of_topics(self):
            import matplotlib.pyplot as plt
            import seaborn as sns
            from matplotlib.lines import Line2D

            plt.figure(figsize=(8,5))

            # Assuming compute_coherence_values has been called
            coherence_values = self.coherence_values
            start = 2
            limit = 40
            step = 6
            x = range(start, limit, step)

            ax = sns.lineplot(x=x, y=coherence_values, color='#238C8C')

            plt.title("Best Number of Topics for LDA Model")
            plt.xlabel("Num Topics")
            plt.ylabel("Coherence score")
            plt.xlim(start, limit)
            plt.xticks(range(2, limit, step))

            plt.axvline(x[np.argmax(coherence_values)], color='#F26457', linestyle='--')

            legend_elements = [Line2D([0], [0], color='#238C8C', ls='-', label='Coherence Value (c_v)'),
                               Line2D([0], [1], color='#F26457', ls='--', label='Optimal Number of Topics')]
            ax.legend(handles=legend_elements, loc='upper right')


    def train_lda_model(self, num_topics):
        """
        Trains the LDA model using the specified number of topics.
        """
        from gensim.models.ldamodel import LdaModel
        self.lda_model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=num_topics, random_state=100, update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    def visualize_topics(self):
        """
        Visualizes the topics using pyLDAvis.
        """
        import pyLDAvis.gensim_models as gensimvis
        import pyLDAvis
        pyLDAvis.enable_notebook()
        lda_viz = gensimvis.prepare(self.lda_model, self.corpus, self.dictionary, sort_topics=True)
        return pyLDAvis.display(lda_viz)

    def add_lda_results_to_dataframe(self, column_name:str):
        """
        Adds LDA model results to the DataFrame: the dominant topic for each document.
        """
        dominant_topics = []
        for tweet_corpus in self.corpus:
            # Get topic distribution for the document
            topic_distribution = self.lda_model.get_document_topics(tweet_corpus)
            # Sort the topics by probability
            topic_distribution = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
            # Get the dominant topic (highest probability) for the document
            dominant_topic = topic_distribution[0][0]  # This is the topic number
            dominant_topics.append(dominant_topic)
        self.dataframe[column_name] = dominant_topics

    def display_topic_words(self, num_words=15, total_topics=None):
           """
           Displays the top words for each topic.

           Parameters:
           - num_words: Number of top words to display for each topic.
           - total_topics: Total number of topics to display. If None, displays all topics.
           """
           total_topics = total_topics or self.lda_model.num_topics  # Use specified or default to model's total
           for idx, topic in self.lda_model.show_topics(num_topics=total_topics, num_words=num_words, formatted=False):
               topic_words = [word for word, _ in topic]
               print(f"Topic: {idx + 1}\nWords: {', '.join(topic_words)}\n")


    def apply_topic_labels(self, topic_labels, input_column:str, output_column:str):
        """
        Applies descriptive labels to topics based on their numeric identifier.
        """
        # Ensure 'dominant_topic' column exists from previous step
        if input_column in self.dataframe.columns:
            self.dataframe[output_column] = self.dataframe[input_column].apply(lambda topic_id: topic_labels.get(topic_id, "Unknown Topic"))
        else:
            print("Dataframe does not have dominant topics. Please run add_lda_results_to_dataframe first.")


class DistilBERTClassifier:
    """
    Manages the training, evaluation, and prediction of a DistilBERT model for topic classification.
    Additionally, it utilizes a pre-trained DistilBERT model for sentiment analysis.
    """

    def __init__(self, dataframe, label_dict):
        self.dataframe = dataframe
        self.label_dict = label_dict
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                                         num_labels=len(label_dict),
                                                                         output_attentions=False,
                                                                         output_hidden_states=False)

    def encode_data(self, text_data, labels):
        """
        Encodes text data into format suitable for DistilBERT.
        """
        encoded_dict = self.tokenizer.batch_encode_plus(
            text_data,
            add_special_tokens=True,
            max_length=256,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        inputs_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        labels = torch.tensor(labels)
        return inputs_ids, attention_masks, labels

    def create_data_loader(self, inputs_ids, attention_masks, labels, batch_size=32, train=True):
        """
        Creates a DataLoader for the training or evaluation dataset.
        """
        data = TensorDataset(inputs_ids, attention_masks, labels)
        if train:
            sampler = RandomSampler(data)
        else:
            sampler = SequentialSampler(data)

        data_loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return data_loader

    def train(self, train_dataloader, validation_dataloader, epochs=4):
        """
        Trains the DistilBERT model.
        """
        import torch

        # Check if a GPU is available and set PyTorch to use the GPU. Otherwise, it uses the CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inform the user which device is being used.
        print(f"Using device: {device}")


        optimizer = AdamW(self.model.parameters(),
                          lr=2e-5,
                          eps=1e-8)

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
                self.model.zero_grad()
                outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask,
                                     labels=b_labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Average training loss: {avg_train_loss}")


    def predict(self, texts):
        """
        Predicts the topics of given texts using the trained DistilBERT model.
        """

        # Check if a GPU is available and set PyTorch to use the GPU. Otherwise, it uses the CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inform the user which device is being used.
        print(f"Using device: {device}")

        self.model.eval()
        predictions = []

        with torch.no_grad():
            inputs_ids, attention_masks, _ = self.encode_data(texts, [0]*len(texts))  # Labels not needed for prediction
            dataloader = self.create_data_loader(inputs_ids, attention_masks, torch.tensor([0]*len(texts)), batch_size=32, train=False)

            for batch in dataloader:
                b_input_ids, b_input_mask = tuple(t.to(device) for t in batch)[:2]
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)

        predicted_labels = [self.label_dict[label] for label in predictions]
        return predicted_labels

    def perform_sentiment_analysis(self, texts):
        """
        Performs sentiment analysis on the provided texts using a separate pre-trained DistilBERT model.
        """
        from transformers import pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")

        results = sentiment_pipeline(texts)
        return results

    def summarize_sentiment_analysis(self, sentiment_results):
        """
        Summarizes and displays the results of sentiment analysis.
        sentiment_results: List of dictionaries with 'label' and 'score' from sentiment analysis.
        """
        from collections import Counter

        # Extract sentiment labels
        sentiments = [result['label'] for result in sentiment_results]
        sentiment_counts = Counter(sentiments)

        # Display summary
        print("Sentiment Analysis Summary:")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} instances")

        for sentiment in sentiment_counts:
            avg_score = sum(result['score'] for result in sentiment_results if result['label'] == sentiment) / sentiment_counts[sentiment]
            print(f"Average confidence for {sentiment}: {avg_score:.2f}")

def display_textual_analysis_results(analysis_results):
    """
    Displays the results of textual analysis including keywords and n-grams for each topic and sentiment.
    analysis_results: A nested dictionary containing topics, sentiments, and their corresponding keywords and n-grams.
    """
    for topic, sentiment_data in analysis_results.items():
        print(f"Topic: {topic}")
        for sentiment, features in sentiment_data.items():
            print(f"  Sentiment: {sentiment}")
            keywords = ", ".join(features['Keywords'])
            bigrams = ", ".join(features['Bigrams'])
            print(f"    Keywords: {keywords}")
            print(f"    Bigrams: {bigrams}\n")
