import json
from exception import EncoderException, RemovePunctuationError

class Data_Preprocessing():
    def json_encoder(self, data):
        if data:
            try:
                with open(data, 'r') as f:
                    encoded_data = json.load(f)
                    return encoded_data
            except:
                raise EncoderException()
    
    def train_test_split(self, X, y):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test


class TextHandler():
    def remove_punctuation(self):
        try:
            import string
            return str.maketrans('', '', string.punctuation)
        except:
            raise RemovePunctuationError()