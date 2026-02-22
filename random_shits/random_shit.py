import pandas as pd 

# df = pd.read_csv('/media/ivsr/manh/oord_data/pose/2021-11-25-12-01-20/imu.csv')
df1 = pd.read_csv('/media/manh/manh/oord_data/pose/2021-11-25-12-01-20/gps.csv')

df = pd.read_csv('/media/manh/manh/mulran/DCC01/gps.csv')
print(len(df))
print(len(df1))
print(len(df.columns))
print(len(df1.columns))