import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    items = file()
    print(items.head())


def file():

    DATA_FOLDER = "C:/Users/Galip/Documents/serdar/harvard/kaggle/competitive-data-science-predict-future-sales"

    transactions = pd.read_csv(os.path.join(DATA_FOLDER, 'sales_train.csv.gz'))
    items = pd.read_csv(os.path.join(DATA_FOLDER, 'items.csv'))
    item_categories = pd.read_csv(os.path.join(DATA_FOLDER, 'item_categories.csv'))
    shops = pd.read_csv(os.path.join(DATA_FOLDER, 'shops.csv'))
    return items


if __name__ == '__main__':
    main()


