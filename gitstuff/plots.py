import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
import sys
#style.use('fivethirtyeight')

df = pd.read_csv(r".\saas_df_v4.csv")
#df['year'] = pd.to_datetime(df['year'], format='%Y')


# use plotly!!!!!!!!!!!!

del df['1 Mo']
del df['3 Mo']
del df['6 Mo']
del df['1 Yr']
del df['3 Yr']
del df['5 Yr']
del df['7 Yr']
del df['20 Yr']
del df['10yr_next']
del df['30yr_next']

# plotting correlation heatmap
#f, ax = plt.subplots(figsize=(10, 8))
#corr = df.corr()
#sns.heatmap(corr,
#    cmap=sns.diverging_palette(220, 10, as_cmap=True),
#    vmin=-1.0, vmax=1.0,
#    square=True, ax=ax)
#plt.show()


# plot a scatterplot of growth
# plotting a scatterplot
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.scatterplot(x='year',
                y='annual_sales_growth_pct', data=df, hue='ipo_year')
ax.set(xlabel ='year', ylabel ='Sales Growth %')
#plt.xlim(df["year"].min(), df["year"].max())
ax.set(xticks=df['year'].unique())
ax.set_xticklabels(labels = df['year'].unique(), rotation=45)
plt.title('Growth')
plt.show()

# plotting a scatterplot of P/S
f, ax = plt.subplots(figsize=(10, 8))
ax = sns.scatterplot(x='year',
                y='price_to_sales', data=df, hue='ipo_year')
ax.set(xlabel ='year', ylabel ='P/S')
ax.set(xticks=df['year'].unique())
ax.set_xticklabels(labels = df['year'].unique(), rotation=45)
plt.title('Price to Sales')
plt.show()


# plot both with yield over time
fig, (ax1,ax2) = plt.subplots(2, figsize=(16,6))
ax1.set_title('P/S')
ax1.set(ylabel ='P/S')
ax1.set(xlabel="")
sns.regplot(x='year', y='price_to_sales', data=df, ax=ax1)

ax2.set_title('30 Yr Yield')
ax2.set(xticks=df['year'].unique())
ax2.set(xlabel ='year', ylabel ='30 yr Treasury Yield %')
ax2.set_xticklabels(labels = df['year'].unique(), rotation=45)
sns.regplot(x='year', y='30 Yr', data=df, ax=ax2)
  
plt.show()


# plot v yield
ax = sns.scatterplot(x='30 Yr',
                y='price_to_sales', data=df, hue='year')
ax.set(xlabel ='30 Yr Yield', ylabel ='P/S')
  
plt.title('Price to Sales vs Treasury Yield')
plt.show()

