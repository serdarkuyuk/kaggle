import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#%matplotlib inline

#from grader import Grader

DATA_FOLDER = '../readonly/final_project_data/'

transactions    = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
items           = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
shops           = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))

transactions['value']=transactions.item_price*transactions.item_cnt_day
#transactions.shape
transactions.head()
#items.shape
#items.head()
#item_categories.shape
#item_categories.head()
#shops.shape
#shops.head()

trans = transactions[transactions['date'].str.endswith('09.2014')]

transactions['day'] = [d.split('.')[0] for d in transactions.date]
transactions['month'] = [d.split('.')[1] for d in transactions.date]
transactions['year'] = [d.split('.')[2] for d in transactions.date]

max_revenue = transactions.query('month == "09" & year == "2014"').groupby(transactions["shop_id"]).sum()["value"].max()

trans = transactions[transactions['date'].str.match('.*0[6-8].2014$')]
trans = trans[trans['date'].str.endswith('12.2014')]

trans= transactions.query('month == ["06", "07", "08"] & year == "2014"')
tel = pd.Series(pd.merge(trans,items,how='left', on="item_id").drop(['day', 'item_name'],axis=1).groupby('item_category_id').sum()['value']).argmax()

category_id_with_max_revenue = tel

a=transactions.groupby(by="item_id").sum() #.sort_values("item_id", axis = 0, ascending = True)
b=transactions.groupby(by="item_id")
t=0
for i in a.index:
    if b.get_group(i).item_price.unique().shape == (1,):
        t += 1

num_items_constant_price = t

shop_id = 25

a=transactions.query('month == "12" & year == "2014" & shop_id == 25 & item_cnt_day != 0').groupby('date').sum()

total_num_items_sold = a.item_cnt_day
days = a.index.to_datetime()

# Plot it
plt.plot(days, total_num_items_sold)
plt.ylabel('Num items')
plt.xlabel('Day')
plt.title("Daily revenue for shop_id = 25")
plt.show()

total_num_items_sold_var = a.loc[:,"item_cnt_day"].var(ddof=1)
