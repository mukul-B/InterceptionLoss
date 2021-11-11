import pandas as pd
import numpy as np
def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    print(a)
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    print(iszero)
    absdiff = np.abs(np.diff(iszero))
    print(absdiff,np.diff(iszero))
    ranges = np.where(absdiff == 1)[0]
    print(ranges)
    return ranges.reshape(-1,2)
# Read CSV file into DataFrame df
df = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-12.expanded.20210324T151954Z.PROVISIONAL/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-12.expanded.20210324T151937Z.csv')
df2 = pd.read_csv('/home/muku/Downloads/NEON_Site_Lat_Long_Biomass.csv')
df3 = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-01.expanded.20210123T023002Z.RELEASE-2021/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-01.expanded.20201016T011717Z.csv')
# Show dataframe
print(df3.keys())
# print(df.keys())
print(df3['TFPrecipExpUncert'].head())
prep=df3['TFPrecipExpUncert'].to_numpy()
# prep=df3['TFPrecipExpUncert'].iloc[22:32].to_numpy()
zero_trail=zero_runs(prep)
print(zero_trail)
for i,j in zero_trail:
    print(i,j)
    print(np.sum(prep[i:j]))

stroms= np.array([[np.sum(prep[i:j])] for i,j in zero_trail])
print("stroms" ,stroms)