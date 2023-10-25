import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

my_dict = {'Milk':[1,0,1,0,1],'Bread':[1,1,1,0,1],'Cheese':[0,1,1,1,0],'Egg':[0,1,1,1,1],'Jam':[1,0,0,0,0]}
df1 = pd.DataFrame(my_dict,index=['trn1','trn2','trn3','trn4','trn5'])
print(df1)
def encode_unit(num):
    if num > 0:
        return True
    else:
        return False
df1 = df1.applymap(encode_unit)
# Apriori Algorithm
print(df1)
my_frequent_item_set = apriori(df1,min_support=0.4,use_colnames=True)
print(my_frequent_item_set)

# Rules Mining Algorithm
my_rules = association_rules(my_frequent_item_set,metric='confidence',min_threshold=0.6)
print(my_rules.to_string())
