import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessing:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
    
    def load_data(self) -> pd.DataFrame:
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def drop_columns(self, columns: list) -> pd.DataFrame:
        self.data = self.data.drop(columns, axis=1)
    
    def correct_data(self) -> pd.DataFrame:
        # loop on every 8 rows
        for i in range(0, len(self.data), 8):
            continue
    
    def correct_customers(self, start: int, end:int) -> pd.DataFrame:
        pass


