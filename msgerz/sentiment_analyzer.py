!pip install transformers
!pip install torch
!pip install pandas
!pip install openpyxl
!pip install vaderSentiment
!pip install textblob
!pip install tr
!pip install googletrans==4.0.0-rc1


import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from afinn import Afinn
from googletrans import Translator
from textblob import TextBlob
from .keyword_sentiment import KeywordSentiment
import sqlite3

class MsgSentiment:
    def __init__(self, keyword=None, keyword_file_path=None, db_path=":memory:"):
        self.keyword = keyword.lower() if keyword else None
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.afinn = Afinn()
        self.translator = Translator()
        self.keyword_sentiment = KeywordSentiment(keyword_file_path) if keyword_file_path else None
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS sentiments (
            keyword TEXT,
            tweet TEXT,
            bert_sentiment TEXT,
            vader_sentiment TEXT,
            afinn_sentiment TEXT,
            textblob_sentiment TEXT,
            keyword_score INTEGER,
            final_sentiment TEXT
        )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_sentiment(self, data):
        query = """
        INSERT INTO sentiments (keyword, tweet, bert_sentiment, vader_sentiment, afinn_sentiment, textblob_sentiment, keyword_score, final_sentiment)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.conn.execute(query, data)
        self.conn.commit()

    def clean_tweet(self, tweet):
        tweet = re.sub(r'http\S+', '', tweet)
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'#\w+', '', tweet)
        tweet = re.sub(r'RT[\s]+', '', tweet)
        tweet = re.sub(r'\n', ' ', tweet)
        tweet = re.sub(r'[^\w\s]', '', tweet)
        return tweet.lower()

    def translate_tweet(self, tweet):
        translated_tweet = self.translator.translate(tweet, dest='en').text
        return translated_tweet

    def analyze_sentiment_bert(self, tweet):
        inputs = self.tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = self.model(**inputs)
        scores = outputs.logits[0].detach().numpy()
        scores = torch.softmax(torch.tensor(scores), dim=0).numpy()

        sentiment_label = 'neutral'
        if scores.argmax() == 4:
            sentiment_label = 'positive'
        elif scores.argmax() == 0:
            sentiment_label = 'negative'

        return sentiment_label

    def analyze_sentiment_vader(self, tweet):
        vs = self.vader_analyzer.polarity_scores(tweet)
        if vs['compound'] > 0.05:
            return 'positive'
        elif vs['compound'] < -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_sentiment_afinn(self, tweet):
        score = self.afinn.score(tweet)
        if score > 0:
            return 'positive'
        elif score < 0:
            return 'negative'
        else:
            return 'neutral'

    def analyze_sentiment_textblob(self, tweet):
        analysis = TextBlob(tweet)
        sentiment_score = analysis.sentiment.polarity

        if sentiment_score > 0:
            return 'positive'
        elif sentiment_score < 0:
            return 'negative'
        else:
            return 'neutral'

    def analyze_keyword_sentiment(self, keyword, tweet):
        cleaned_tweet = self.clean_tweet(tweet)
        translated_tweet = self.translate_tweet(cleaned_tweet)
        bert_sentiment = self.analyze_sentiment_bert(translated_tweet)
        vader_sentiment = self.analyze_sentiment_vader(translated_tweet)
        afinn_sentiment = self.analyze_sentiment_afinn(translated_tweet)
        textblob_sentiment = self.analyze_sentiment_textblob(translated_tweet)
        keyword_score = self.keyword_sentiment.calculate_score(keyword, cleaned_tweet) if self.keyword_sentiment else 0
        return {
            'bert': bert_sentiment,
            'vader': vader_sentiment,
            'afinn': afinn_sentiment,
            'textblob': textblob_sentiment,
            'keyword_score': keyword_score
        }

    def process_excel(self, file_path):
        data = pd.read_excel(file_path)
        sentiments = []
        bert_sentiments = []
        vader_sentiments = []
        afinn_sentiments = []
        textblob_sentiments = []
        keyword_scores = []

        for index, row in data.iterrows():
            keyword = row['keyword']
            tweet = row['tweet']
            sentiment = self.analyze_keyword_sentiment(keyword, tweet)
            bert_sentiment = sentiment['bert']
            vader_sentiment = sentiment['vader']
            afinn_sentiment = sentiment['afinn']
            textblob_sentiment = sentiment['textblob']
            keyword_score = sentiment['keyword_score']
            bert_sentiments.append(bert_sentiment)
            vader_sentiments.append(vader_sentiment)
            afinn_sentiments.append(afinn_sentiment)
            textblob_sentiments.append(textblob_sentiment)
            keyword_scores.append(keyword_score)
            final_sent = self.final_sentiment(bert_sentiment, vader_sentiment, afinn_sentiment, textblob_sentiment, keyword_score)
            sentiments.append(final_sent)
            self.insert_sentiment((keyword, tweet, bert_sentiment, vader_sentiment, afinn_sentiment, textblob_sentiment, keyword_score, final_sent))

        data['bert_sentiment'] = bert_sentiments
        data['vader_sentiment'] = vader_sentiments
        data['afinn_sentiment'] = afinn_sentiments
        data['textblob_sentiment'] = textblob_sentiments
        data['keyword_score'] = keyword_scores
        data['final_sentiment'] = sentiments
        output_file_path = file_path.replace('.xlsx', '_with_sentiments.xlsx')
        data.to_excel(output_file_path, index=False)
        print(f"Results are saved to {output_file_path}")

        # Ekrana yazdırma
        print(data[['keyword', 'tweet', 'bert_sentiment', 'vader_sentiment', 'afinn_sentiment', 'textblob_sentiment', 'keyword_score', 'final_sentiment']])

    def final_sentiment(self, bert_sentiment, vader_sentiment, afinn_sentiment, textblob_sentiment, keyword_score):
        sentiments = [bert_sentiment, vader_sentiment, afinn_sentiment, textblob_sentiment]
        if sentiments.count('positive') >= 3 or (sentiments.count('positive') == 2 and keyword_score > 0):
            return 'positive'
        elif sentiments.count('negative') >= 3 or (sentiments.count('negative') == 2 and keyword_score < 0):
            return 'negative'
        elif sentiments.count('neutral') >= 3 or (sentiments.count('neutral') == 2 and keyword_score == 0):
            return 'neutral'
        else:
            return 'N/A'
