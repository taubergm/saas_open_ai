import pandas as pd

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv(r".\saas_df_v4.csv")
#x_cols = ['FedFundsRate', '30 Yr', 'annual_sales_growth_pct', 'quarter_sales_growth_pct', 'ps_prev']
x_cols = ['ps_prev', '30 Yr', 'annual_sales_growth_pct']

df = df.dropna(subset = x_cols)

X = df[x_cols]
y = df['price_to_sales']


############# polynomial
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=1/5, random_state=1)
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_poly, y)
y_pred = poly_reg_model.predict(X_test)

# model evaluation
print(
  'mean_squared_error : ', mean_squared_error(y_test, y_pred))
print(
  'mean_absolute_error : ', mean_absolute_error(y_test, y_pred))


new_df = pd.DataFrame()
new_df['y'] = y_test
new_df['predict'] = y_pred

print(new_df)