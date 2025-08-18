import pandas as pd

stock_prices = pd.read_excel('SPX_database_2010.xlsx', sheet_name='prices')
mkt_cap = pd.read_excel('SPX_database_2010.xlsx', sheet_name='mkt_cap')

stock_prices.to_parquet("stock_prices.parquet", index=False, engine="pyarrow")
mkt_cap.to_parquet("mkt_cap.parquet", index=False, engine="pyarrow")