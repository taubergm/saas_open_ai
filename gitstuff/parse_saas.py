####
####
### NEED TO ROUND DATES TO NEAREST WEEK for merge
####
#### OR BETTER - for EFF AND TREASURIES FILL IN MISSING DATES WITH LAST VAL
###


import os     
import re  
import pandas as pd   
import numpy as np
import calendar
from datetime import datetime, timedelta
import time
import sys
import matplotlib.pyplot as plt
import seaborn as sns




def ParsePSFiles(path):
    files = [os.path.join(PATH, f)  for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
    txt_files = [f for f in files if f.endswith('txt')]

    saasDf = pd.DataFrame()
    ps_dicts = []

    for t in txt_files:

        first_match = True

        with open(t) as f:

            for line in f:

                ps_dict = {}
                file_name = os.path.basename(t)
                ps_dict['ticker'] = re.sub('.txt', '', file_name)

                m = re.search("(.*)\s+P/S", line)
                if m is not None: # get company name 
                    ps_dict['name'] = m.group(1)

                if re.search("^20", line):
                    
                    results = line.split()

                    if len(results) == 3:
                        ps_dict['date'] = results[0]
                        ps_dict['stock_price'] = results[1]
                        ps_dict['price_to_sales'] = results[2]
                    elif len(results) == 4:
                        ps_dict['date'] = results[0]
                        ps_dict['stock_price'] = results[1]
                        ps_dict['price_to_sales'] = results[3]


                    ps_dicts.append(ps_dict.copy()) 

    df = pd.DataFrame()
    df = pd.DataFrame.from_dict(ps_dicts, orient='columns') 
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['price_to_sales'] = df['price_to_sales'].astype('float')
    df = df.sort_values(['ticker', 'date'], ascending=[True,True])
    
    return df


def RoundDates(df):
    # round date to nearest week
    df['date_seconds'] = ((df['date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype(int)
    threshhold = 111333
    threshhold = 100 # then get unique values 
    df['interval'] = round(df['date_seconds'] / threshhold) * threshhold
    # Want to calculate percent change per ticker per date
    # first ticker value will be invalid so remove it

def AddMissingDates(df):
    # I know this is inefficent, but df is small
    # fill in missing dates with identical data
    ## MAKE THIS A SEPERATE FUNCTION

    df = df.sort_values(['date'], ascending=[True]).reset_index()

    new_rows = []
    prev_row = {}
    for index, row in df.iterrows():
        new_rows.append(row)
        if index == 0:
            prev_row = row
            continue

        delta = row['date'] - prev_row['date'] 
        if delta.days > 1: # there are missing vales
            #print(row['date'], prev_row['date'])
            delta = timedelta(days=1)
            date_idx = prev_row['date'] + delta
            while (date_idx < row['date']):
                new_row = row.copy()
                new_row['date'] = date_idx
                date_idx = date_idx + delta
                new_rows.append(new_row)
            
        prev_row = row

    df = pd.DataFrame.from_dict(new_rows, orient='columns') # better way to build df
    
    return df


def GetEFFData(path):
    eff_df = pd.read_csv(r'C:\users\v-mtauberg\saas\eff_rate.csv')
    eff_df['date'] = pd.to_datetime(eff_df['Effective Date'], format='%m/%d/%Y')
    eff_df = AddMissingDates(eff_df)
    
    # round date to nearest week
    #eff_df['date_seconds'] = ((eff_df['Effective Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')).astype(int)
    #eff_df['interval'] = round(eff_df['date_seconds'] / threshhold) * threshhold

    return eff_df


def GetTreasuryData(path):
    files = [os.path.join(PATH, f)  for f in os.listdir(PATH) if os.path.isfile(os.path.join(PATH, f))]
    treasury_files = [f for f in files if re.search("daily-treasury", f)]

    treasury_df = pd.concat((pd.read_csv(f) for f in treasury_files), ignore_index=True)
    treasury_df['date'] = pd.to_datetime(treasury_df['Date'], format='%m/%d/%Y')

    treasury_df.to_csv(r"C:\Users\v-mtauberg\saas\treasury_df.csv", index=False)

    treasury_df = AddMissingDates(treasury_df)

    return treasury_df



#########
## Get P/S data
#########
PATH = r'C:\Users\v-mtauberg\saas'
df = ParsePSFiles(PATH)
print(f"Len of orig df = {len(df)}")

##############################################
## Add Federal Funds Rate
##############################################
PATH = r"C:\Users\v-mtauberg\saas\eff_rate.csv"
eff_df = GetEFFData(PATH)
df = df.merge(eff_df, left_on='date', right_on='date')
#df = df.merge(eff_df, left_on='interval', right_on='interval')
df = df.sort_values(['ticker', 'date'], ascending=[True, True]).reset_index()
print(f"Len of new df = {len(df)}")

########
## Add US Treasuries
#######
PATH = r'C:\Users\v-mtauberg\saas'
treasury_df = GetTreasuryData(PATH)

print(treasury_df)
print(treasury_df.info())

print(df['date'].head(5))
print(treasury_df['date'].head(5))

df = df.merge(treasury_df, left_on='date', right_on='date')
print(f"Len of new df = {len(df)}")


###################
# Process % changes#
###################


#df['ps_prev'] = df['price_to_sales'].shift(1)
#df['ps_prev']= df['ps_prev'].where(df.duplicated(subset = ['ticker']))
#df['ps_pct_change'] = (df['price_to_sales'] - df['ps_prev']) / df['ps_prev']*100
#df['ps_pct_change'] = df['ps_pct_change'].where(df.duplicated(subset = ['ticker']))

# add sales

df['stock_price'] = df['stock_price'].astype('float')
df['sales'] = df['stock_price']/df['price_to_sales']

df['sales_prev'] = df['sales'].shift(1)
df['sales_prev']= df['sales_prev'].where(df.duplicated(subset = ['ticker']))
df['sales_growth_pct'] = (df['sales'] - df['sales_prev']) / df['sales_prev']*100
df['sales_growth_pct'] = df['sales_growth_pct'].where(df.duplicated(subset = ['ticker']))


df = df.sort_values(['ticker', 'date'], ascending=[True,True])
#df['eff_prev'] = df['FedFundsRate'].shift(1)
#df['eff_prev']= df['eff_prev'].where(df.duplicated(subset = ['ticker']))
df['FedFundsRateNext'] = df['FedFundsRate'].shift(-1)
#df['FedFundsRateNext']= df['FedFundsRateNext'].where(df.duplicated(subset = ['ticker']))
#df['eff_pct_past_change'] = (df['FedFundsRate'] - df['eff_prev']) / df['eff_prev']*100
#df['eff_pct_past_change'] = df['eff_pct_past_change'].where(df.duplicated(subset = ['ticker']))
#df['eff_past_change'] = (df['FedFundsRate'] - df['eff_prev'])
#df['eff_past_change'] = df['eff_past_change'].where(df.duplicated(subset = ['ticker']))
#df['eff_pct_future_change'] = (df['FedFundsRate'] - df['eff_next']) / df['eff_next']*100
#df['eff_pct_future_change'] = df['eff_pct_future_change'].where(df.duplicated(subset = ['ticker']))
#df['eff_future_change'] = (df['FedFundsRate'] - df['eff_next'])
#df['eff_future_change'] = df['eff_future_change'].where(df.duplicated(subset = ['ticker']))


#df['10yr_prev'] = df['10 Yr'].shift(1)
#df['10yr_prev']= df['10yr_prev'].where(df.duplicated(subset = ['ticker']))
df['10yr_next'] = df['10 Yr'].shift(-1)
df['30yr_next'] = df['30 Yr'].shift(-1)
#df['10yr_next']= df['10yr_next'].where(df.duplicated(subset = ['ticker']))
#df['10yr_pct_past_change'] = (df['10 Yr'] - df['10yr_prev']) / df['10yr_prev']*100
#df['10yr_pct_past_change']= df['10yr_pct_past_change'].where(df.duplicated(subset = ['ticker']))
#df['10yr_pct_future_change'] = (df['10 Yr'] - df['10yr_next']) / df['10yr_next']*100
#df['10yr_pct_future_change']= df['10yr_pct_future_change'].where(df.duplicated(subset = ['ticker']))
#df['10yr_future_change'] = (df['10 Yr'] - df['10yr_next'])
#df['10yr_future_change']= df['10yr_future_change'].where(df.duplicated(subset = ['ticker']))

df['year'] = pd.DatetimeIndex(df['date']).year

############################
# Normalize Cols of Interest
###########################
# https://stackoverflow.com/questions/46419180/pandas-normalize-within-the-group
#df['ticker_norm_ps'] = df['price_to_sales'] / df.groupby('ticker')['price_to_sales'].transform('sum')
# should do x-xmin/xmax-xmin
#df['ticker_norm_ps'] = (df['price_to_sales'] - df.groupby('ticker')['price_to_sales'].transform('min')) / (df.groupby('ticker')['price_to_sales'].transform('max') - df.groupby('ticker')['price_to_sales'].transform('min'))


#### Extra Stuff #
df = df.drop(['index_x', 'index_y', 'level_0'], axis=1)
df = df.drop(['Target Rate To (%)', 'Target Rate From (%)'], axis=1)
df = df.drop(['2 Mo', '4 Mo'], axis=1)
df = df.drop(['Effective Date', 'Date', 'Rate Type'], axis=1)


#df = df.drop(['stock_price'], axis=1)

#####
#add age of company!!!!!!!!!
####

print(df.columns)
df.to_csv(r"C:\users\v-mtauberg\saas\saas_df.csv")



################
# Explore Correlations and Analysys
#################

#print(df.corr())
df = pd.read_csv(r".\saas_df_V3.csv")
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