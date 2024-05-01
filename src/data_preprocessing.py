import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
            # print progress bar
            print(f"\rProgress: {i+8}/{len(self.data)}", end="")

    
    def correct_columns_type(self):
        # first convert to string then remove any non-numeric characters
        self.data['Age'] = self.data['Age'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Age'] = pd.to_numeric(self.data['Age'], errors='coerce')

        self.data['Annual_Income'] = self.data['Annual_Income'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Annual_Income'] = pd.to_numeric(self.data['Annual_Income'], errors='coerce')


        self.data['Changed_Credit_Limit'] = self.data['Changed_Credit_Limit'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Changed_Credit_Limit'] = pd.to_numeric(self.data['Changed_Credit_Limit'], errors='coerce')

        self.data['Amount_invested_monthly'] = self.data['Amount_invested_monthly'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Amount_invested_monthly'] = pd.to_numeric(self.data['Amount_invested_monthly'], errors='coerce')


        # self.data['Type_of_Loan'] = self.data['Type_of_Loan'].astype(str).str.replace(r'and', '', regex=True)

        # self.data['Credit_History_Age'] = self.data['Credit_History_Age'].astype(str)

        self.data['Num_of_Delayed_Payment'] = self.data['Num_of_Delayed_Payment'].astype(str).str.replace(r'[^0-9\.\-]', '', regex=True)
        self.data['Num_of_Delayed_Payment'] = pd.to_numeric(self.data['Num_of_Delayed_Payment'], errors='coerce')



    def correct_customers(self, start: int, end:int):
        # self.correct_month(start, end)                                                              # 1 Month
        self.correct_age(start, end)                                                                # 2 Age
        self.correct_occupation(start, end)                                                         # 3 Occupation
        self.correct_continuous('Annual_Income', start, end)                                        # 4 Annual_Income
        self.correct_continuous('Monthly_Inhand_Salary', start, end)                                # 5 Monthly_Inhand_Salary
        self.correct_categorical('Num_Bank_Accounts', start, end)                                   # 6 Num_Bank_Accounts
        self.correct_categorical('Num_Credit_Card', start, end)                                     # 7 Num_Credit_Card
        # TODO: check if this is correct , have one value per customer
        self.correct_categorical('Interest_Rate', start, end)                                       # 8 Interest_Rate
        self.correct_categorical('Num_of_Loan', start, end)                                            # 9 Num_of_Loan
        # 10 Type_of_Loan
        self.correct_delay_from_due_date(start, end) # 11 Delay_from_due_date
        self.correct_num_of_delayed_payment(start, end) # 12 Num_of_Delayed_Payment

        # TODO: check it's float column
        self.correct_continuous('Changed_Credit_Limit',start,end) # 13 Changed_Credit_Limit
        self.correct_num_credit_inquiries(start, end) # 14 Num_Credit_Inquiries
        # TODO: get Mode of all Data and replace with it as default value
        self.correct_categorical('Credit_Mix', start, end,'Standard')                               # 15 Credit_Mix
        # TODO: check it's float column
        self.correct_continuous('Outstanding_Debt', start, end)                                     # 16 Outstanding_Debt
        # TODO: check it's float column
        self.correct_continuous('Credit_Utilization_Ratio', start, end)                             # 17 Credit_Utilization_Ratio
        # self.correct_Credit_History_Age(start, end)                                                 # 18 Credit_History_Age
        # TODO: get Mode of all Data and replace with it as default value
        self.correct_categorical('Payment_of_Min_Amount', start, end, 'No')                         # 19 Payment_of_Min_Amount
        self.correct_continuous('Total_EMI_per_month', start, end)                                  # 20 Total_EMI_per_month
        # TODO: check it's float column
        self.correct_continuous('Amount_invested_monthly', start, end)                              # 21 Amount_invested_monthly
        # TODO: get Mode of all Data and replace with it as default value
        self.correct_categorical('Payment_Behaviour', start, end,'High_spent_Small_value_payments') # 22 Payment_Behaviour
        self.correct_continuous('Monthly_Balance', start, end) # 23 Monthly_Balance
        
    
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
                self.data.at[i,'month'] = months[month_counter]
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
                self.data.at[i,'Occupation'] = most_frequent_occupation
            #TODO: check if this is correct maybe someone changed the occupation
            elif self.data['Occupation'][i] != most_frequent_occupation:
                self.data.at[i,'Occupation'] = most_frequent_occupation

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
                self.data.at[i,'Monthly_Inhand_Salary'] = mean_salary
        return None
    
    def correct_Credit_History_Age(self, start: int, end:int):
        '''
        Replace missing values in the "Credit History Age" column with the mean of the column within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        years = 0
        months = 0
        index = 0
        for i in range(start, end):
            if "Years" in self.data['Credit_History_Age'][i]:
                # get the number of years
                years = int(self.data['Credit_History_Age'][i].split()[0])
                months = int(self.data['Credit_History_Age'][i].split()[3]) 
                index = i
                break
        
        for i in range(start, end):
            if self.data['Credit_History_Age'][i] == "NA" or pd.isnull(self.data['Credit_History_Age'][i]):
                self.data.at[i,'Credit_History_Age'] = f"{years} Years {months + i - index} Months"
                
        return None
    
    def correct_type_of_loan(self, start: int, end:int):
        '''
        Replace missing values in the "Type of Loan" column with the most frequent type of loan within the specified range.
        
        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        pass
        # # replace missing values with most frequent type of loan
        # try:
        #     # get most frequent type of loan
        #     most_frequent_loan = self.data['Type_of_Loan'][start:end].mode(dropna=True)[0]

        #     if pd.isna(most_frequent_loan) :
        #         print("NNNNNNNNNNNNN")

        # except:
        #     # if no mode, replace with empty string
        #     # No loans Taken
        #     most_frequent_loan = ''

        # print("most_frequent_loan",most_frequent_loan)

        # for i in range(start, end):
        #     if pd.isnull(self.data['Type_of_Loan'][i]):
        #         self.data.loc[i,'Type_of_Loan'] = most_frequent_loan

        # return None
    
    def correct_delay_from_due_date(self, start: int, end:int):
        '''

        Delay_from_due_date ==> Represents the average number of days delayed from the payment date

        Replace missing values in the column with the mean(int) of the column within the specified range.
        Fixes negative values by setting them to 0.

        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).

        Returns: None
        '''
        # Replace missing values with mean of the column
        try:
            # replace missing values with mean of the column
            mean = int(self.data['Delay_from_due_date'][start:end].mean(skipna=True))
        except:
            # if no mean, replace with 0
            mean = 0
        
        for i in range(start, end):
            if pd.isnull(self.data['Delay_from_due_date'][i]):
                self.data.at[i,'Delay_from_due_date'] = mean
            if self.data['Delay_from_due_date'][i] < 0:
                self.data.at[i,'Delay_from_due_date'] = 0
     
        return None

    def correct_num_of_delayed_payment(self, start: int, end:int):
        '''
        Num_of_Delayed_Payment ==> Represents the average number of payments delayed by a person

        Replace missing values in the column with the mean(int) of the column within the specified range.
        Fixes negative values by setting them to 0.

        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).

        Returns: None
        '''
        # Replace missing values with mean of the column
        try:
            # replace missing values with mean of the column
            mean = int(self.data['Num_of_Delayed_Payment'][start:end].mean(skipna=True))
        except:
            # if no mean, replace with 0
            mean = 0

        for i in range(start, end):
            if pd.isnull(self.data['Num_of_Delayed_Payment'][i]):
                self.data.at[i,'Num_of_Delayed_Payment'] = mean
            if self.data['Num_of_Delayed_Payment'][i] < 0:
                self.data.at[i,'Num_of_Delayed_Payment'] = 0
        return None    
    
    def correct_num_credit_inquiries(self, start: int, end:int):
        '''
        Num_Credit_Inquiries ==> Represents the number of credit inquiries made by a person

        Replace missing values in the column with the mod of the column within the specified range.

        Parameters:
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).

        Returns: None
        '''
        # replace missing values with mode of the column
        try:
            # get mode of the column
            most_frequent = self.data['Num_Credit_Inquiries'][start:end].mode(dropna=True)[0]
        except:
            # if no mode, replace with 0
            most_frequent = 0

        for i in range(start, end):
            if pd.isnull(self.data['Num_Credit_Inquiries'][i]):
                self.data.at[i,'Num_Credit_Inquiries'] = most_frequent

        pass

        

    def correct_categorical(self,column ,start: int, end:int,default_value=0):
        '''
        Replace missing values in the "Number of Bank Accounts" column with the mod of the column within the specified range.
        
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
            most_frequent = default_value

        for i in range(start, end):
            if pd.isnull(self.data[column][i]):
                self.data.at[i,column] = most_frequent
            #TODO: check if this is correct, maybe someone changed the number of bank accounts
            elif self.data[column][i] != most_frequent:
                self.data.at[i,column] = most_frequent
        return None
    
    def correct_continuous(self,column ,start: int, end:int,default_value=0):
        '''
        Replace missing values in the column with the mean of the column within the specified range.
        
        Parameters:
        column (str): The column to correct.
        start (int): The starting index to begin correction.
        end (int): The ending index to end correction (exclusive).
        
        Returns: None
        '''
        # Replace missing values with mean of the column
        try:
            # replace missing values with mean of the column
            mean = self.data[column][start:end].mean(skipna=True)
        except:
            # if no mean, replace with 0
            mean = default_value

        for i in range(start, end):
            if pd.isnull(self.data[column][i]):
                self.data.at[i,column] = mean
        return None    
    def convert_Y_to_numerical(self):
        # convert Y to numerical
        # Method 1: Label Encoding
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(self.data['Credit_Score'])
        
    def convert_catgories_to_numerical(self):
        # convert categorical columns to numerical
        # Method 2: One-Hot Encoding

        # Columns to transform (same as used during training)
        categorical_cols = ['Occupation', 'Interest_Rate', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

        # Columns to leave unchanged (same as used during training)
        continous_cols = [ 'Age','Num_Bank_Accounts','Num_Loan','Interest_Rate','Num_Credit_Card','Annual_Income', 'Monthly_Inhand_Salary', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

        # Define the same ColumnTransformer from training
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continous_cols),  # Scale numerical columns
                ('cat', OneHotEncoder(), categorical_cols)  # One-hot encode categorical columns
            ],
            remainder='passthrough'  # Leave other columns unchanged
        )
        return self.preprocessor.fit_transform(self.data)
    
if __name__ == '__main__':
    data_preprocessing = DataPreprocessing('dataset/train.csv')
    data_preprocessing.load_data()

    data_preprocessing.drop_columns(['ID','Customer_ID','Name','SSN','Month','Credit_History_Age'])

    data_preprocessing.correct_columns_type()

    data_preprocessing.correct_data()

    data_preprocessing.save_data('dataset/train_preprocessed.csv')

    # data_preprocessing.correct_age(0, 8)

    # data_preprocessing.correct_delay_from_due_date(0, 8)
    # data_preprocessing.correct_num_credit_inquiries(0, 8)
    # data_preprocessing.correct_type_of_loan(35,40)
    # print(data_preprocessing.data['Type_of_Loan'][35:40])


    # print(data_preprocessing.data['Num_of_Delayed_Payment'][138:145])
    # print(data_preprocessing.correct_num_of_delayed_payment(0,8))




    # print(data_preprocessing.get_data()['Age'][0:8])



# python src/data_preprocessing.py
