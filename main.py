import numpy as np
import pandas as pd


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def strom_event(zero_trail, storm_len, storm_gap):
    diffin = np.array([zero_trail[i + 1][0] - zero_trail[i][1] for i in range(len(zero_trail) - 1)] + [-1])
    empty_array = np.empty((0, 2), int)
    start, end = zero_trail[0]
    for i, j in zip(zero_trail, diffin):
        end = i[1]
        if j > storm_gap:
            if end - start > storm_len:
                empty_array = np.append(empty_array, np.array([[start, end]]), axis=0)
            start = end + j
        elif j == -1:
            if i[1] - i[0] > storm_len:
                empty_array = np.append(empty_array, np.array([[i[0], i[1]]]), axis=0)
    return empty_array


def agg_prec(df, prec, tf):
    df['startDateTime'] = pd.to_datetime(df['startDateTime'])
    prep = df[prec].to_numpy()
    # prep=df3['TFPrecipExpUncert'].iloc[22:32].to_numpy()
    zero_trail = zero_runs(prep)
    df_res = pd.DataFrame(columns=['startDateTime', 'duration', 'p','t', 'IL'])
    storms = strom_event(zero_trail, 3, 6)
    for i, j in storms:
        # print(i, j)
        df_res = df_res.append(
            {'startDateTime': df['startDateTime'][i],
             'duration': pd.Timedelta(df['startDateTime'][j - 1] - df['startDateTime'][i]).seconds / 60.0,
             'p': df[prec][i:j].sum(),
             't':df[tf][i:j].sum(),
             'IL':(df[prec][i:j].sum()-df[tf][i:j].sum()) / df[prec][i:j].sum()*100} , ignore_index=True)
    return df_res



if __name__ == "__main__":
    # Read CSV file into DataFrame df
    site = 'ABBY' #NEON Site name here
    ID = 'D16' #put ID of NEON site here
    mo = '2020-11' #change year and month here

    biomass_df = pd.read_csv('resource/NEON_Site_Lat_Long_Biomass.csv')
    lai_df = pd.read_csv('resource/LAI-500m-8d-MCD15A2H-006-results.csv')

    prec_df = pd.read_csv(
        'resource/NEON.' + ID + '.' + site + '.DP1.00006.001.000.050.030.SECPRE_30min.' + mo + '.expanded.20210324T151136Z.csv')
    thrfall_df = pd.read_csv(
        'resource/NEON.' + ID + '.' + site + '.DP1.00006.001.001.000.030.THRPRE_30min.' + mo + '.expanded.20210324T151136Z.csv')

    print(prec_df.keys())
    print(biomass_df.keys())
    print(thrfall_df.keys())
    print(lai_df.keys())

    newdf = biomass_df.loc[(biomass_df.Site == site)]
    prec_df["tf"]=thrfall_df["TFPrecipBulk"]
    agg_prec_df = agg_prec(prec_df, 'secPrecipBulk', 'tf')
    #agg_thrfall_df = agg_prec(thrfall_df, 'TFPrecipBulk','t')

    print(agg_prec_df.head()) # 19
    #print(agg_thrfall_df.he0ad()) #14
    # need to make it same
    #interception=pd.concat([agg_prec_df, agg_thrfall_df], axis=1)
    interception=agg_prec_df
    interception['Site'] = site
    #print(interception[["startDateTime", "duration","Site"]].head())
    #print("interception")
    interception= pd.merge(interception,newdf,on="Site")
    #print(interception[['startDateTime', 'duration','p','t']].head())

    print(interception[["Site", 'Mean Canopy Height (m)', 'USFS Forest Biomass (mg/ha)']].head())
    #print(biomass_df.head())
