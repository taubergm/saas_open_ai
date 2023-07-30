import pandas as pd
from datetime import datetime, timedelta



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


def GetCPIData():
    df = pd.read_csv(r'C:\users\v-mtauberg\saas\cpi_data.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = AddMissingDates(df)
    print(df)

    return df

def GetM2Data():
    df = pd.read_csv(r'C:\users\v-mtauberg\saas\M2SLMoney.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    df = AddMissingDates(df)

    return df



df = pd.read_csv(r".\saas_df_v5.csv")
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
print(df)

cpi_df = GetCPIData()
m2_df = GetM2Data()

cpi_df.to_csv('test_cpi.csv', index=False)

df = df.merge(cpi_df, left_on='date', right_on='date')
df = df.merge(m2_df, left_on='date', right_on='date')



# add ps_prev
#df['ps_prev']  = df['price_to_sales'].shift(1)
#df['ps_prev']= df['ps_prev'].where(df.duplicated(subset = ['ticker']))
#df['ipo_year'] = df.groupby('ticker')['year'].transform('min')
#df['year_from_ipo'] = df['year'] - df['ipo_year']
#df['quarterly_growth_rolling_avg'] = df['quarter_sales_growth_pct'].rolling(window=4).mean()
#df = df.dropna(subset = 'annual_sales_growth_pct')
#print(df[['ticker', 'price_to_sales', 'ps_prev', 'year_from_ipo']])





df.to_csv(r"C:\users\v-mtauberg\saas\saas_df_v6.csv", index=False)
