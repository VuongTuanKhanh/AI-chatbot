class Data_Handler():

    def extract(self, path):
        import os
        if os.path.exists(self.original_path) and len(os.listdir(self.path)) == 2:
            import zipfile
            with zipfile.ZipFile(self.original_path, 'r') as zip_data:
                zip_data.extractall(self.path)

    def json_encoder(self, intent, data):
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
    
    @property
    def original_path(self):
        return f'{self.path}/data.zip'


    @property
    def extracted_data_path(self):
        return f'{self.path}/data'