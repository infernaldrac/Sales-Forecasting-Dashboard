import pandas as pd
from pathlib import Path
p = Path('Dataset/Sample - Superstore.csv')
print('exists', p.exists())
df = pd.read_csv(p)
print('columns', df.columns.tolist())
print('order date sample', df['Order Date'].head().tolist())
print('order dtype', df['Order Date'].dtype)
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
print('dtype after', df['Order Date'].dtype)
print('na count', df['Order Date'].isna().sum())
print('sample after parse', df['Order Date'].head().tolist())
print('sales sample dtype', df['Sales'].dtype)
print('sales sample', df['Sales'].head().tolist())
print('group attempt')
try:
    monthly = df.dropna(subset=['Order Date']).groupby(pd.Grouper(key='Order Date', freq=pd.offsets.MonthEnd()))['Sales'].sum().reset_index()
    print(monthly.head())
except Exception as e:
    import traceback
    traceback.print_exc()
