import os
from core.exception import ExtractionException

class Data_Handler():

    def __init__(self, path=''):
        self.path = path

    def extract(self):
        import os
        if os.path.exists(self.original_path) and len(os.listdir(f'{self.path}/data')) == 1:
            try:
                import zipfile
                with zipfile.ZipFile(self.original_path, 'r') as zip_data:
                    zip_data.extractall(self.path)
            except:
                raise ExtractionException()
        return self.extracted_data_path
    
    @property
    def original_path(self):
        return f'{self.path}/data/original.zip'


    @property
    def extracted_data_path(self):
        return f'{self.path}/data'
