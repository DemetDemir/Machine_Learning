import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
sns.set(style='whitegrid', context='talk', palette='rainbow')

PATH = 'Kaggle_Tutorials\melb_data.csv'

melbourne_data = pd.read_csv(PATH)
#print(melbourne_data.describe())

print(melbourne_data.sort_values(by='YearBuilt'))


