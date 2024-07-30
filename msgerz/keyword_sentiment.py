import pandas as pd

class KeywordSentiment:
    def __init__(self, file_path):
        self.positive_keywords = {}
        self.negative_keywords = {}
        self.load_keywords(file_path)

    def load_keywords(self, file_path):
        data = pd.read_excel(file_path)
        for _, row in data.iterrows():
            keyword = row['keyword'].lower()
            positives = row['positive'].split(',') if pd.notna(row['positive']) else []
            negatives = row['negative'].split(',') if pd.notna(row['negative']) else []
            self.positive_keywords[keyword
