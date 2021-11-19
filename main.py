import pandas as pd
import numpy as np

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0]
    return ranges.reshape(-1,2)

def agg_prec(df3):
    prep = df3['TFPrecipBulk'].to_numpy()
    # prep=df3['TFPrecipExpUncert'].iloc[22:32].to_numpy()
    zero_trail = zero_runs(prep)
    df8 = pd.DataFrame(columns=['duration', 'throughfall'])
    for i, j in zero_trail:
        # print(i, j)
        df8 = df8.append(
            {'duration': pd.Timedelta(df3['startDateTime'][j - 1] - df3['startDateTime'][i]).seconds / 60.0,
             'throughfall': df3['TFPrecipBulk'][i:j].sum()}, ignore_index=True)
    return df8

if __name__=="__main__":
    # Read CSV file into DataFrame df
    prec_df = pd.read_csv('resource/NEON.D16.ABBY.DP1.00006.001.000.050.030.SECPRE_30min.2020-11.expanded.20210324T151136Z.csv')
    biomass_df = pd.read_csv('resource/NEON_Site_Lat_Long_Biomass.csv')
    thrfall_df = pd.read_csv('resource/NEON.D16.ABBY.DP1.00006.001.001.000.030.THRPRE_30min.2020-11.expanded.20210324T151136Z.csv')
    lai_df = pd.read_csv('resource/LAI-500m-8d-MCD15A2H-006-results.csv')

    print(thrfall_df.keys())
    thrfall_df['startDateTime'] = pd.to_datetime(thrfall_df['startDateTime'])
    storm_df= agg_prec(thrfall_df)
    print(storm_df.head())

