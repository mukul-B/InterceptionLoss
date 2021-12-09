import glob
import pandas as pd

#this program reads in all throughfall files per site and date, calculates the average (exculding any all-0 files), and writes a csv file with the startdatetime and avg throughfall values.

def tfavg(site, date):
    file_list = glob.glob('resource/NEON.D*.' + site + '.DP1.00006.*.000.030.THRPRE_30min.' + date + '.basic.*.csv')
    l = len(file_list)
    TF = pd.DataFrame()
    i=0
    j=1
    for f in file_list:
        tf_df = pd.read_csv(f, usecols=['TFPrecipBulk'])
        tf_df.rename(columns={'TFPrecipBulk': 'TF_' + str(j)}, inplace=True)
        #only include TF columns that are not empty
        #if tf_df['TF_'+str(j)].sum() > 0:
        TF.insert(i, 'TF_'+str(j), tf_df['TF_'+str(j)])
        j=j+1
        i=i+1
        #else:
        #    j=j+1
        #    print(site, date, 'TF_', j, 'no values')
    # average all columns of TF readings
    TF['average']=TF.mean(axis=1)
    #include start date column to match with precip data in main code
    TF['startdateTime'] = pd.read_csv(file_list[0], usecols=['startDateTime'])

    #add only avg and start date to new dataframe
    df_tf = pd.DataFrame()
    df_tf['startdateTime'] = TF['startdateTime']
    df_tf['TFPrecipBulkAvg'] = TF['average']
    #write files
    df_tf.to_csv('resource/AvgTF/TFavg.'+site+'.'+date+'.csv', index=False)

site_list = ['BART', 'BLAN', 'SCBI', 'SERC', 'DSNY', 'JERC', 'OSBS', 'GUAN', 'STEI', 'TREE', 'UNDE', 'KONZ',
             'UKFS', 'GRSM', 'MLBS', 'ORNL', 'DELA', 'LENO', 'TALL', 'RMNP', 'CLBJ', 'YELL', 'SRER', 'ABBY',
             'WREF', 'SJER', 'SOAP', 'TEAK', 'BONA', 'JORN', 'DEJU']

dates = ['2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09',
             '2020-10', '2020-11', '2020-12']#, '2021-01', '2021-02', '2021-03', '2021-04', '2021-05', '2021-06',
             #'2021-07', '2021-08', '2021-09', '2021-10']

#dates = range(1,13)
for site in site_list:
    for date in dates:
        #date = "2020-" + str(date).rjust(2, '0')
        tfavg(site, date)
        print(site, date)

print('Program completed.')

