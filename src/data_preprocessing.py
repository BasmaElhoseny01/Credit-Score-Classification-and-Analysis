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
    
    def save_data(self, path: str):
        self.data.to_csv(path, index=False)
    
    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def drop_columns(self, columns: list):
        self.data = self.data.drop(columns, axis=1)
    
    def correct_data(self) -> pd.DataFrame:
        # loop on every 8 rows
        for i in range(0, len(self.data), 8):
            self.correct_customers(i, i+8)
    
    def correct_columns_type(self):
        # first convert to string then remove any non-numeric characters
        self.data['Age'] = self.data['Age'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Age'] = pd.to_numeric(self.data['Age'], errors='coerce')

        self.data['Annual_Income'] = self.data['Annual_Income'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Annual_Income'] = pd.to_numeric(self.data['Annual_Income'], errors='coerce')

    def correct_customers(self, start: int, end:int):
        self.correct_month(start, end)
        self.correct_occupation(start, end)
        self.correct_age(start, end)
        self.correct_int_categorical('Num_Bank_Accounts', start, end)
        self.correct_int_categorical('Num_Credit_Card', start, end)
        self.correct_int_categorical('Num_Loan', start, end)
        # TODO: check if this is correct , have one value per customer
        self.correct_int_categorical('Interest_Rate', start, end)
        self.correct_int_continuous('Annual_Income', start, end)
        self.correct_int_continuous('Monthly_Inhand_Salary', start, end)

    
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

        try:
            # get most frequent occupation
            most_frequent_occupation = self.data['Occupation'][start:end].mode(dropna=True)[0]
        except:
            # if no mode, replace with 'Other'
            most_frequent_occupation = '_______'

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

        # convert age to int
        try:
            # get most frequent age
            most_frequent_age = self.data['Age'][start:end].mode(dropna=True)[0]

            if most_frequent_age < 0:
                # try second most frequent age if exists
                if len(self.data['Age'][start:end].mode(dropna=True)) > 1:
                    most_frequent_age = self.data['Age'][start:end].mode(dropna=True)[1]
                    if most_frequent_age < 0:
                        most_frequent_age = 0
        except:
            #TODO: check if this is correct, may replace with mean of age of all customers
            # if no mode, replace with 0
            most_frequent_age = 0

        for i in range(start, end):
            if pd.isnull(self.data['Age'][i]):
                self.data.at[i, 'Age'] = most_frequent_age
            elif self.data['Age'][i] != most_frequent_age:
                self.data.at[i, 'Age'] = most_frequent_age
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
    
    def correct_int_categorical(self,column ,start: int, end:int):
        '''
        Replace missing values in the "Number of Bank Accounts" column with the mean of the column within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        # replace missing values with mode of the column
        try:
            # get mode of the column
            most_frequent = self.data[column][start:end].mode(dropna=True)[0]
        except:
            # if no mode, replace with 0
            most_frequent = 0

        for i in range(start, end):
            if pd.isnull(self.data[column][i]):
                self.data[column][i] = most_frequent
            #TODO: check if this is correct, maybe someone changed the number of bank accounts
            elif self.data[column][i] != most_frequent:
                self.data[column][i] = most_frequent
        return None
    
    def correct_int_continuous(self,column ,start: int, end:int):
        '''
        Replace missing values in the column with the mean of the column within the specified range.
        
        Parameters:
        column (str): The column to correct.
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        try:
            # replace missing values with mean of the column
            mean = self.data[column][start:end].mean(skipna=True)
        except:
            # if no mean, replace with 0
            mean = 0

        for i in range(start, end):
            if pd.isnull(self.data[column][i]):
                self.data[column][i] = mean
        return None    


if __name__ == '__main__':
    data_preprocessing = DataPreprocessing('dataset/train.csv')
    data_preprocessing.load_data()
    data_preprocessing.correct_columns_type()
    data_preprocessing.correct_age(0, 8)

    print(data_preprocessing.get_data()['Age'][0:8])



# python src/data_preprocessing.py
