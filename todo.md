NB: Handling Missing Values and -ve values are handled according to teh feature type

### Data Studying

1. Each 12 Consecutive records are for one Customer during 12 months (Jan-Dec)

#### ID:

#### Customer_ID:

#### Name:
1. has some missing values --> get from other records with same customer_ID

#### Age:
1. -ve Values --> get value from other records(12)
2. Some has _ (Need cleaning) (ex:30_) --> remove \_(special chars) or get from other records(12)
3. OverFlowed Values (ex: 70120) -->

#### SSN:
1. Needs cleaning (ex:#F%$D@\*&12) --> get from other records(12)

#### Occupation:
1. Missing Values or (**\_**) --> get from other records(12)

#### Annual Income:
1. values with special characters such as (\_) --> Remove Special Character or Take average from other records(10)

#### Monthly_Inhand_Salary:
1. Missing Values --> Take average from other records(10)

#### Num_Bank_Accounts:
1. OverFlowed Values (ex:1414) --> Take Values from others(10)

### Num_Credit_Card:
1. OverFlowed Values (ex:1414) --> Take Values from others(10)

Interest_Rate:

1. OverFlowed Values (ex:1414) --> Take Values from others(10)

Num_of_Loan:

1. negative values --> Take Values from others(10)
2. OverFlowed Values (ex:1414) --> Take Values from others(10)
3. Values with special characters such as (\_) --> Remove Special Character or Take average from other records(10)

Type_of_Loan: [String]
Need to be split and converted to numerical values

1. Missing Values --> Take Values from others(10)
2. Not Specified [loan type] --> to be checked

### Delay_from_due_date:

1. negative values --> Take Most frequent value from other records(8)
2. nan: replaced by most frequent value [Not this isn't present in data but we handled it]

### Num_of_Delayed_Payment:

1. Missing Values -->
2. negative values -->
3. Values with special characters such as (\_) -->

Changed_Credit_Limit:

1. Missing Values -->
2. negative values -->
3. Values with special characters such as (\_) -->

Num_Credit_Inquiries:

1. Missing Values -->
2. overFlowed Values (ex:1050) -->

Credit_Mix:

1. Missing Values or (\_) -->

Outstanding_Debt:

Credit_Utilization_Ratio:
Remove (')

Credit_History_Age: [String]

1. NA -->

Payment_of_Min_Amount:

1. NM -->

Total_EMI_per_month:

Amount_invested_monthly:

Payment_Behaviour:

1. Rubbish Values ex:!@9#%8 --> Take Average from other records(10)

Monthly_Balance:

Credit_Score:

[]
[] AWS or Azure
