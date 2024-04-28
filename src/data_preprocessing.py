import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

class DataPreprocessing:
    def __init__(self, data_path: str) -> None:
        self.data_path = data_path
    
    def load_data(self):
        self.data = pd.read_csv(self.data_path)
        return self.data
    
    def drop_columns(self, columns: list):
        self.data = self.data.drop(columns, axis=1)
    
    def correct_data(self) -> pd.DataFrame:
        # loop on every 8 rows
        for i in range(0, len(self.data), 8):
            continue
    
    def correct_customers(self, start: int, end:int):
        pass
    
    def correct_month(self, start: int, end:int):
        '''
        Corrects the "Month" column in the DataFrame from the starting index to the ending index.

        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).

        Returns: None
        '''
        # from start to end, correct the month column
        months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        month_counter = 0
        for i in range(start, end):
            # check if month is correct place in the list
            if self.data['Month'][i] != months[month_counter]:
                self.data['month'][i] = months[month_counter]
            month_counter += 1

        return None
    
    def correct_occupation(self, start: int, end:int):
        '''
        Replace missing values in the "Occupation" column with the most frequent occupation within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        # replace missing values with most frequent occupation

        # get most frequent occupation
        most_frequent_occupation = self.data['Occupation'][start:end].mode()[0]

        # replace missing values with most frequent occupation
        for i in range(start, end):
            if pd.isnull(self.data['Occupation'][i]):
                self.data['Occupation'][i] = most_frequent_occupation
            #TODO: check if this is correct maybe someone changed the occupation
            elif self.data['Occupation'][i] != most_frequent_occupation:
                self.data['Occupation'][i] = most_frequent_occupation

        return None
    
    def correct_age(self, start: int, end:int):
        '''
        Replace missing values in the "Age" column with the most frequent age within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''

        # get most frequent occupation
        most_frequent_age = self.data['Age'][start:end].mode()[0]

        # check if most frequent age is null
        if pd.isnull(most_frequent_age):
            #TODO: check if this is correct, may replace with mean of age of all customers
            most_frequent_age = 0 

        for i in range(start, end):
            if pd.isnull(self.data['Age'][i]):
                self.data['Age'][i] = most_frequent_age
            elif self.data['Age'][i] != most_frequent_age:
                self.data['Age'][i] = most_frequent_age
        return None
        
    def correct_monthly_inhand_salary(self, start: int, end:int):
        '''
        Replace missing values in the "Monthly In-hand Salary" column with the mean of the column within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        # replace missing values with mean of the column
        mean_salary = self.data['Monthly_Inhand_Salary'][start:end].mean()

        for i in range(start, end):
            if pd.isnull(self.data['Monthly_Inhand_Salary'][i]):
                self.data['Monthly_Inhand_Salary'][i] = mean_salary
        return None



# # Test 
# data_preprocessing = DataPreprocessing('./dataset/train.csv')
# data_preprocessing.load_data()
# data_preprocessing.correct_monthly_inhand_salary(0,8)