import numpy as np
import pandas as pd
import pandasql as ps
import matplotlib.pyplot as plt
import seaborn as sns



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


def agg_prec(df, prec, type):
    df['startDateTime'] = pd.to_datetime(df['startDateTime'])
    prep = df[prec].to_numpy()
    # prep=df3['TFPrecipExpUncert'].iloc[22:32].to_numpy()
    zero_trail = zero_runs(prep)
    df_res = pd.DataFrame(columns=[type + 'startDateTime', type + 'duration', type])
    storms = strom_event(zero_trail, 3, 6)
    for i, j in storms:
        # print(i, j)
        df_res = df_res.append(
            {type + 'startDateTime': df['startDateTime'][i],
             type + 'duration': pd.Timedelta(df['startDateTime'][j - 1] - df['startDateTime'][i]).seconds / 60.0,
             type: df[prec][i:j].sum()}, ignore_index=True)
    return df_res

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
    site = 'ABBY'

    biomass_df = pd.read_csv('resource/static/NEON_Site_Lat_Long_Biomass.csv')
    lai_df = pd.read_csv('resource/static/LAI-500m-8d-MCD15A2H-006-results.csv')

    prec_df = pd.read_csv(
        'resource/NEON.D16.' + site + '.DP1.00006.001.000.050.030.SECPRE_30min.2020-11.expanded.20210324T151136Z.csv')
    thrfall_df = pd.read_csv(
        'resource/NEON.D16.' + site + '.DP1.00006.001.001.000.030.THRPRE_30min.2020-11.expanded.20210324T151136Z.csv')

    print(prec_df.keys())
    print(biomass_df.keys())
    print(thrfall_df.keys())
    # print(lai_df.keys())

    site_biomass = biomass_df.loc[(biomass_df.Site == site)]
    site_lai = lai_df.loc[lai_df.Category == site]

    prec_df["tf"] = thrfall_df["TFPrecipBulk"]
    # agg_prec_df = agg_prec(prec_df, 'secPrecipBulk', 'tf')
    # agg_thrfall_df = agg_prec(thrfall_df, 'TFPrecipBulk', 't')

    # print(agg_prec_df.head()) # 19
    # print(agg_thrfall_df.head()) #14
    # need to make it same
    # interception = pd.concat([agg_prec_df, agg_thrfall_df], axis=1)
    interception=agg_prec(prec_df, 'secPrecipBulk', 'tf')
    interception['Site'] = site

    interception = pd.merge(interception, site_biomass, on="Site")

    interception["Date"] = interception["startDateTime"].dt.date
    interception["ldiFromDate"] = interception["Date"] - pd.Timedelta("3 day")
    interception["ldiToDate"] = interception["Date"] + pd.Timedelta("4 day")

    # A.Date,A.ldiFromDate,A.ldiToDate,B.Date,
    sqlcode = '''
    select B.MCD15A2H_006_Lai_500m
    from interception A
    left outer join site_lai B on A.Site=B.Category
    where A.ldiFromDate <= B.Date and A.ldiToDate >= B.Date
    '''

    Lai_500m = ps.sqldf(sqlcode, locals())
    print(Lai_500m)
    interception["Lai_500m"]= Lai_500m
    print(interception.keys())
    interception_loss_df= interception[["Lai_500m","Mean Canopy Height (m)","USFS Forest Biomass (mg/ha)","duration","p","IL"]]
    print(interception_loss_df.head())
    plt.close()
    # plt.matshow(interception_loss_df.corr())
    sns.set_style("whitegrid")

    # Create the pairplot with the first 100 values in our subset of data
    # Color each sample according to its value in the Target column
    # The height of each plot will be 2 inches
    sns.pairplot(interception_loss_df, hue="IL", height=2)
    plt.show()
