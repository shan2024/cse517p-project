import os
import pandas as pd

class NGramDataLoader():
    def __init__(self):
        print("Making NGramEmbedder")
        self.test_data = None
        self.dev_data = None
        self.train_data = None
    
    def load(self, data_directory):

        files = [f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.test_data = pd.DataFrame()
        self.dev_data = pd.DataFrame()
        self.train_data = pd.DataFrame()

        for file_name in files:
            data = pd.read_csv(f"{data_directory}/{file_name}")["dialogue"]

            if "dev" in file_name:
               self.dev_data = pd.concat([self.dev_data, data], ignore_index=True)
            if f"test" in file_name:
                self.test_data = pd.concat([self.test_data, data], ignore_index=True)
            if "train" in file_name:
                self.train_data = pd.concat([self.train_data, data], ignore_index=True)
        
        self.test_data  = self.test_data.sample(frac=1).reset_index(drop=True)
        self.dev_data = self.dev_data.sample(frac=1).reset_index(drop=True)
        self.train_data = self.train_data.sample(frac=1).reset_index(drop=True)