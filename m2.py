import glob

import numpy as np
import pandas as pd
import pandasql as ps


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def strom_event(zero_trail, storm_len, storm_gap):
    diffin = np.array([zero_trail[i + 1][0] - zero_trail[i][1] for i in range(len(zero_trail) - 1)] + [-1])
    empty_array = np.empty((0, 2), int)
    if len(zero_trail) == 0:
        return empty_array
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
    df_res = pd.DataFrame(columns=['startDateTime', 'duration', 'p', 't', 'IL'])
    storms = strom_event(zero_trail, 3, 6)
    for i, j in storms:
        # print(i, j)
        prep = df[prec][i:j].sum()
        thr = df[tf][i:j].sum()
        itcp_loss = 0 if prep == 0 else ((prep - thr) / prep) * 100

        if prep > 5 and thr!=0:
            df_res = df_res.append(
                {'startDateTime': df['startDateTime'][i],
                 'duration': pd.Timedelta(df['startDateTime'][j - 1] - df['startDateTime'][i]).seconds / 60.0,
                 'p': prep,
                 't': thr,
                 'IL': itcp_loss}, ignore_index=True)
    return df_res





def staging(precip_path, thrfall_path, site, prec_type):
    biomass_df = pd.read_csv('resource/static/Biomass.csv')
    lai_df = pd.read_csv('resource/static/LAI-500m-8d-MCD15A2H-006-results.csv')
    # prec_df = pd.read_csv(
    #     'resource/NEON.D16.' + site + '.DP1.00006.001.000.050.030.SECPRE_30min.2020-11.expanded.20210324T151136Z.csv')
    # thrfall_df = pd.read_csv(
    #     'resource/NEON.D16.' + site + '.DP1.00006.001.001.000.030.THRPRE_30min.2020-11.expanded.20210324T151136Z.csv')
    prec_df = pd.read_csv(precip_path)
    thrfall_df = pd.read_csv(thrfall_path)

    # print(prec_df.keys())
    # print(biomass_df.keys())
    # print(thrfall_df.keys())
    # print(lai_df.keys())

    site_biomass = biomass_df.loc[(biomass_df.Site == site)]
    site_lai = lai_df.loc[lai_df.Category == site]

    prec_df["tf"] = thrfall_df["TFPrecipBulkAvg"]
    # agg_prec_df = agg_prec(prec_df, 'secPrecipBulk', 'tf')
    # agg_thrfall_df = agg_prec(thrfall_df, 'TFPrecipBulk', 't')

    # print(agg_prec_df.head()) # 19
    # print(agg_thrfall_df.head()) #14
    # need to make it same
    # interception = pd.concat([agg_prec_df, agg_thrfall_df], axis=1)
    interception = agg_prec(prec_df, prec_type + 'PrecipBulk', 'tf')
    interception['Site'] = site
    print(len(interception) )
    if len(interception) != 0:
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
        # print(Lai_500m)
        interception["Lai_500m"] = Lai_500m


        # print(interception.keys())
        # columns_list = ('startDateTime', 'duration', 'p', 't', 'Site', 'IL', 'Biomass', 'MCH', 'LAI')
        #
        interception_loss_df = interception[
            ["startDateTime","Lai_500m", "MCH", "Biomass", "duration", "p", "IL"]]
    # print(interception_loss_df.head())
        interception_loss_df.to_csv('C:/Users/Abigail Sandquist/PycharmProjects/InterceptionLoss/resource/Staging/output2.csv', mode='a', header=False, index=False)


if __name__ == "__main__":

    Sites = ['BART', 'HARV', 'BLAN', 'SCBI', 'SERC', 'DSNY', 'JERC', 'OSBS', 'GUAN', 'STEI', 'TREE', 'UNDE', 'KONZ',
             'UKFS', 'GRSM', 'MLBS', 'ORNL', 'DELA', 'LENO', 'TALL', 'RMNP', 'CLBJ', 'YELL', 'SRER', 'ABBY',
             'WREF', 'SJER', 'SOAP', 'TEAK', 'BONA', 'JORN', 'DEJU']
    # Sites = ['ABBY']
    #
    dates = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09',
             '2020-10', '2020-11', '2020-12', '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
             '2021-07', '2021-08', '2021-09', '2021-10']

    #dates = range(1,13)
    for site in Sites:
        for date in dates:
            #date="2020-"+str(date).rjust(2, '0')
            print(site, date)

            precip_path = glob.glob(
                'resource/NEON.D*.' + site + '.DP1.00006.001.*.SECPRE_30min.' + date + '.basic.*.csv')

            prec_type = "sec"
            if (len(precip_path) == 0):
                precip_path = glob.glob(
                    'resource/NEON.D*.' + site + '.DP1.00006.001.*.PRIPRE_30min.' + date + '.basic.*.csv')
                prec_type = "pri"

            thrfall_path = glob.glob(
                'resource/AvgTF/TFAvg.' + site + '.' + date + '.csv')

            if len(precip_path) == 0 or len(thrfall_path) == 0:
                print("file does not exist")
            else:
                staging(precip_path[0], thrfall_path[0], site, prec_type)
