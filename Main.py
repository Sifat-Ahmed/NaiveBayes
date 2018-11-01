import Naive_Bayes as NB
import pandas as pd
from sklearn.model_selection import train_test_split

#Reading the data
df = pd.read_csv('banknote.csv', index_col=False)
#print(df.describe())

# Splitting the data with test size = 0.2
# Meaning 20% test data
train_data, test_data = train_test_split(df , test_size = 0.2 , shuffle=True)

#print(test_data.head())

GNB = NB.GaussianNaiveBayes()
GNB.train(train_data)
GNB.test(test_data)
print('Accuracy' , GNB.accuracy)
