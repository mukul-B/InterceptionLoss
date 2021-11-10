import pandas as pd

# Read CSV file into DataFrame df
df = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-12.expanded.20210324T151954Z.PROVISIONAL/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-12.expanded.20210324T151937Z.csv')
df2 = pd.read_csv('/home/muku/Downloads/NEON_Site_Lat_Long_Biomass.csv')
df3 = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-01.expanded.20210123T023002Z.RELEASE-2021/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-01.expanded.20201016T011717Z.csv')
# Show dataframe
print(df3.keys())
# print(df.keys())
print(df3['TFPrecipBulk'].head())
print("some command or something")