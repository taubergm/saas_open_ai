import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
#from keras.models import Sequential
#from keras.layers import Dense, Conv1D, Flatten


#https://www.kaggle.com/code/amar09/regression-algorithms-using-scikit-learn
# Try CNN later  - https://www.datatechnotes.com/2019/12/how-to-fit-regression-data-with-cnn.html

df = pd.read_csv(r".\saas_df_v6.csv")
print(df)
cor = df.corrwith(df['price_to_sales']).sort_values(ascending=True)
print(cor)

# plotting correlation heatmap
f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr,
    cmap=sns.diverging_palette(220, 10, as_cmap=True),
    vmin=-1.0, vmax=1.0,
    square=True, ax=ax)
plt.show()

# plotting a scatterplot
ax = sns.scatterplot(x='year',
                y='price_to_sales', data=df)
ax.set(xlabel ='year', ylabel ='P/S')
  
plt.title('Price to Sales')
plt.show()

sys.exit()

print(df.columns)
x_cols = ['30 Yr', 'quarter_sales_growth_pct', 'annual_sales_growth_pct', 'ps_prev']

#x_cols = ['FedFundsRate', '30 Yr', 'year', 'annual_sales_growth_pct']

df = df.dropna(subset = x_cols)



X = df[x_cols]
y = df['price_to_sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=101)

# Normalize!!! -> try normalizing within ticker? ie no hard/fast rule about P/S but we can model change with rates
mm = make_pipeline(MinMaxScaler(), Normalizer())
X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)


# creating a regression model
model = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)

# fitting the model
model.fit(X_train,y_train)


# making predictions
predictions = model.predict(X_test)

new_df = pd.DataFrame()
new_df['y'] = y_test
new_df['predict'] = predictions

print(new_df)
# model evaluation
print(
  'mean_squared_error : ', mean_squared_error(y_test, predictions))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, predictions))

