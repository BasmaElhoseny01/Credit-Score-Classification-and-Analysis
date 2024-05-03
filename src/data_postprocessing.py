import pandas as pd
import numpy as np
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DataPostprocessing:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
    
    def save_data(self, path: str):
        self.data.to_csv(path, index=False)
    
    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def drop_columns(self, columns: list):
        self.data = self.data.drop(columns, axis=1)

    def count_unique_loans(self):
        unique_loans = set()  # Initialize an empty set to store unique loan types
        
        # Iterate through the data
        for i in range(len(self.data)):
            # Split the string into a list of loan types
            loans = self.data.at[i, 'Type_of_Loan'].split(',')
            
            # Add each loan type to the set
            for loan in loans:
                if loan=='nan':
                    continue
                unique_loans.add(loan.strip())  # Remove leading/trailing whitespaces
                
            # Print progress bar
            print(f"\rProgress: {i+1}/{len(self.data)}", end="")

        unique_loans = [loan.strip() for loan in unique_loans]
        # Print the count of unique loans
        print(f"\nNumber of unique loans: {len(unique_loans)}")

        return unique_loans,len(unique_loans)
    
    def correct_columns_type(self):
        # first convert to string then remove any non-numeric characters
        pass

    def correct_data(self) -> pd.DataFrame:
        self.unique_loans,self.count_loans=self.count_unique_loans()

        # loop on every 8 rows
        for i in range(0, len(self.data), 8):
            self.correct_customers(i, i+8)
            # print progress bar
            print(f"\rProgress: {i+8}/{len(self.data)}", end="")

        self.post_processing()
        
        

    def correct_customers(self, start: int, end:int):
        self.correct_month(start, end)                                                               # 1 Month
        
        
    
    def correct_month(self, start: int, end:int):
        pass
     
    
   
    def post_processing(self):
        pass

     

if __name__ == '__main__':
    data_preprocessing = DataPostprocessing('dataset/train_preprocessed.csv')
    data_preprocessing.load_data()

    data_preprocessing.correct_columns_type()

    data_preprocessing.correct_data()
    data_preprocessing.save_data('dataset/train_postprocessed.csv')

     

# python src/data_postprocessing.py
