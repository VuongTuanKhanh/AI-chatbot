class BaseException(Exception):
    def __init__(self, message='Error'):
        self.message = message
    def __str__(self):
        return self.message

class ExtractionException(BaseException):
    def __init__(self, message='No extraction file found'):
        self.message = message


class EncoderException(BaseException):
    def __init__(self, message='No encoder format found'):
        self.message = message

class PreprocessingException(BaseException):
    def __init__(self, message='Error in data preprocessing'):
        self.message = message

class RemovePunctuationError(BaseException):
    def __init__(self, message='Remove punctuation fail'):
        self.message = message