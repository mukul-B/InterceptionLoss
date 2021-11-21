import numpy as np
import pandas as pd


def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.not_equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def storm_event(zero_trail, storm_len, storm_gap):
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
    storms = storm_event(zero_trail, 3, 6)
    for i, j in storms:
        # print(i, j)
        if df[prec][i:j].sum()>5:
            df_res = df_res.append(
            {'startDateTime': df['startDateTime'][i],
             'duration': pd.Timedelta(df['startDateTime'][j - 1] - df['startDateTime'][i]).seconds / 60.0,
             'p': df[prec][i:j].sum(),
             't':df[tf][i:j].sum(),
             'IL':(df[prec][i:j].sum()-df[tf][i:j].sum()) / df[prec][i:j].sum()*100} , ignore_index=True)
    return df_res



if __name__ == "__main__":
    # Read File_Name CSV file into DataFrame df_Names
    df_Names = pd.read_csv('resource/File_Names2020.csv')

    # site = 'ABBY' #NEON Site name here
    # ID = 'D16' #put ID of NEON site here
    # mo = '2020-11' #change year and month here

    #Read in biomass and LAI files
    biomass_df = pd.read_csv('resource/Biomass.csv')
    lai_df = pd.read_csv('resource/LAI-500m-8d-MCD15A2H-006-results.csv')
    columns_list = ('startDateTime', 'duration', 'p', 't', 'Site', 'IL', 'Biomass', 'MCH', 'LAI')
    final_df = pd.DataFrame(columns = columns_list)

    for a in range(30): #change range to run through all files. (note an indexing error happens at range 33, i don't know why)
        #Read in precipitation and throughfall files
        prec_df = pd.read_csv('resource/'+ df_Names.iloc[a,0])
        thrfall_df = pd.read_csv('resource/'+ df_Names.iloc[a,1])
        file = df_Names.iloc[a,0] #store file name into string called file
        file_tf = df_Names.iloc[a,1]
        site = file[9:13] #pull site name out of file name
        #print("files: ", file, "\n", file_tf) #print statement to check file name
        #print("site:", site) #print statement to check site name

        #print(prec_df.keys())
        #print(biomass_df.keys())
        #print(thrfall_df.keys())
        #print(lai_df.keys())

        if file[40:43] == 'SEC':
            P_Name = 'secPrecipBulk'
        elif file[40:43] == 'PRI':
            P_Name = 'priPrecipBulk'


        newdf = biomass_df.loc[(biomass_df.Site == site)]
        prec_df['tf'] = thrfall_df['TFPrecipBulk']
        agg_prec_df = agg_prec(prec_df, P_Name, 'tf')

        #print(agg_prec_df.head())  # 19
        interception = agg_prec_df
        interception['Site'] = site #add site to all storm rows in interception df
        interception = pd.merge(interception, newdf, on="Site") #merge with biomass df on site
        #print(interception[["Site", 'Mean Canopy Height (m)', 'USFS Forest Biomass (mg/ha)']].head())
        final_df = pd.concat([final_df, interception], ignore_index=True, sort=False)
        #print("iteration: ", a)

    print("Program completed")
    print(final_df.head())
    #print(final_df)

