import pandas as pd
import numpy as np

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0]
    return ranges.reshape(-1,2)

def agg_prec(df3):
    prep = df3['TFPrecipExpUncert'].to_numpy()
    # prep=df3['TFPrecipExpUncert'].iloc[22:32].to_numpy()
    zero_trail = zero_runs(prep)
    df8 = pd.DataFrame(columns=['duration', 'throughfall'])
    for i, j in zero_trail:
        # print(i, j)
        df8 = df8.append(
            {'duration': pd.Timedelta(df3['startDateTime'][j - 1] - df3['startDateTime'][i]).seconds / 60.0,
             'throughfall': df3['TFPrecipExpUncert'][i:j].sum()}, ignore_index=True)
    return df8

if __name__=="__main__":
    # Read CSV file into DataFrame df
    df = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-12.expanded.20210324T151954Z.PROVISIONAL/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-12.expanded.20210324T151937Z.csv')
    df2 = pd.read_csv('/home/muku/Downloads/NEON_Site_Lat_Long_Biomass.csv')
    df3 = pd.read_csv('/home/muku/Downloads/NEON_precipitation/NEON.D16.ABBY.DP1.00006.001.2020-01.expanded.20210123T023002Z.RELEASE-2021/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-01.expanded.20201016T011717Z.csv')
    print(df3.keys())
    df3['startDateTime'] = pd.to_datetime(df3['startDateTime'])
    storm_df= agg_prec(df3)
    print(storm_df.head())

