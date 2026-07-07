import pandas as pd
import os
paths = ['Sample - Superstore.csv', 'Dataset/Sample - Superstore.csv', 'Superstore.csv', 'Dataset/Superstore.csv']
for p in paths:
    print('PATH', p, os.path.exists(p))
    if os.path.exists(p):
        df = pd.read_csv(p)
        print('columns', df.columns.tolist())
        print('shape', df.shape)
        if 'Order Date' in df.columns:
            print('Order Date sample', df['Order Date'].head().tolist())
            df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True, errors='coerce')
            print('converted dtype', df['Order Date'].dtype)
            print(df['Order Date'].head())
            print('NaT count', df['Order Date'].isna().sum())
            try:
                monthly = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum().reset_index()
                print('monthly head', monthly.head())
            except Exception as e:
                print('groupby error', type(e).__name__, e)
        break
