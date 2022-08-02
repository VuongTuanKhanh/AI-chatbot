import os, json
import pandas as pd
from exception import EncoderException, PreprocessingException
from data_preprocessing import Data_Preprocessing, TextHandler

class AVI_data_preprocessing(Data_Preprocessing, TextHandler):
    def json_encoder(self, intent, data):
        try:
            batch_data = []
            for e in data:
                parts = []
                for p in e["data"]:
                    parts.append(p["text"])
                batch_data.append({
                    "intent": intent,
                    "text": "".join(parts)
                })
            return batch_data
        except:
            raise EncoderException()


    def train_test_split(self, path):
            # Define train and test json data
            train_json_data = []
            test_json_data = []

            # Loop to get the data
            for intent_name in next(os.walk(path))[1]:
                train_file = f"{path}/{intent_name}/train_{intent_name}_full.json"
                test_file = f"{path}/{intent_name}/validate_{intent_name}.json"

                with open(train_file, "r", encoding="utf8") as f:
                    train_json = json.load(f)
                    train_json_data += self.json_encoder(intent_name, train_json[intent_name])
                    
                with open(test_file, "r", encoding="utf8") as f:
                    test_json = json.load(f)
                    test_json_data += self.json_encoder(intent_name, test_json[intent_name])
                
                self.train_df = pd.DataFrame.from_records(train_json_data).sample(frac=1)
                self.test_df = pd.DataFrame.from_records(test_json_data).sample(frac=1)

            return (self.train_df, self.test_df)

    def text_handle(self):
        try:
            self.train_df["text"] = self.train_df["text"].apply(lambda text: text.lower().translate(self.remove_punctuation()))
            self.test_df["text"] = self.test_df["text"].apply(lambda text: text.lower().translate(self.remove_punctuation()))

            return (self.train_df, self.test_df)
        except:
            raise PreprocessingException()

    
    def vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Create the vectorizer
        vectorizer = TfidfVectorizer()

        # fit and transform the train feature
        X_train = vectorizer.fit_transform(self.train_df['text'])
        # transform the train label
        y_train = self.train_df['intent']

        # fit and transform the test feature
        X_test = vectorizer.transform(self.test_df['text'])
        # transform the test label
        y_test = self.test_df['intent']

        return X_train, y_train, X_test, y_test