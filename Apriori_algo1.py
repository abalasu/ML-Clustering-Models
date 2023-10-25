import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

my_retail_data = pd.read_excel('d:/pythondata/Online_Retail.xlsx')
print(my_retail_data.head())
print(my_retail_data.shape)
my_retail_data = my_retail_data.where(my_retail_data['Country'] == 'Germany')
my_retail_data['Description'] = my_retail_data['Description'].str.strip() # Remove blanks within the description field
my_retail_data.dropna(axis=0,subset=['InvoiceNo'], inplace=True) # Drop any unfilled invoice numbers
my_retail_data['InvoiceNo'] = my_retail_data['InvoiceNo'].astype('str') # Convert InvoiceNo to a string
my_retail_data = my_retail_data[~my_retail_data['InvoiceNo'].str.contains('C')] # Remove Credit Transactions
print(my_retail_data.shape)
print(my_retail_data['Country'].value_counts())

mybasket = (my_retail_data[my_retail_data['Country']=='Germany']
            .groupby(['InvoiceNo','Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))
print(mybasket.head())

def encode_unit(num):
    if num > 0:
        return 1
    else:
        return 0

mybasket_set = mybasket.applymap(encode_unit)
mybasket_set.drop('POSTAGE', inplace=True, axis=1)

my_frequent_item_set = apriori(mybasket_set,min_support=0.07,use_colnames=True)
print(my_frequent_item_set.shape)
my_rules = association_rules(my_frequent_item_set,metric='lift',min_threshold=1)
print(my_rules.shape)

my_final_set = my_rules[(my_rules['lift']>=3) & (my_rules['confidence']>=0.2)]
print(str(my_final_set))