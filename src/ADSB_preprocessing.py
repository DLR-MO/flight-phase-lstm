'''
Author: Emy Arts (emy.arts@dlr.de)

Preprocessing and quality statement of ADSB data
'''

import numpy as np
import pandas as pd
import os
from random import sample
import matplotlib.pyplot as plt
from copy import deepcopy
import argparse
import re
import math
from sklearn.cluster import DBSCAN
from typing import List
import warnings
warnings.filterwarnings("ignore")



TUDE_NAME = {"time": "ts", "icao24": "icao", "lat": "lat", "lon": "lon", "baroaltitude": "alt", "geoaltitude": "alt_backup",\
 "velocity":"spd", "heading": "hdg", "vertrate":"roc"}

HYPER_PARAMETERS = {
    'cut_off_percentiles': {'alt': 0.97, 'alt_backup': 0.97, 'spd': 0.98, 'other': 0.99},
    'extra_outlier_median_window': 25,
    'extra_outlier_smoothing_window': 11,
    'cluster_distance': 300,
    'cluster_threshold': 75,
    'filling_distance_threshold': 5
}

M_WDW = HYPER_PARAMETERS['extra_outlier_median_window']
S_WDW = HYPER_PARAMETERS['extra_outlier_smoothing_window']
C_DIST = HYPER_PARAMETERS['cluster_distance']
C_CUT = HYPER_PARAMETERS['cluster_threshold']
CUT_OFF_PERC = HYPER_PARAMETERS['cut_off_percentiles']
F_TH = HYPER_PARAMETERS['filling_distance_threshold']

### Import and initial pre processing ###

def alpha_numeric_sort(l:List[str])-> list:
    '''
    Sort the given list in the way that humans expect.
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def get_df_list(dirname:list ='csv', naming_convention: dict =TUDE_NAME, flight_numbering:bool =True, excl_index:List[int] = [], number_of_samples:int =None, flight:int =None):
    '''
    Imports csv files to a list of dataframes, where each dataframe is ordered chronologically.
    Attention the flight number should be right before the ".csv".

    :param dirname: the path to the directory of the csv files.
    :param naming_convention: a python dictionary where the keys are the columns that should be considered and the values their desired names.
    :param flight_numbering: whether or not to add a column that keeps track of what flight number it is (from 0 to number_samples).
    :param excl_index: A list of indices to exclude if the filenames are numbered (the index is the position of the file in alphabetic order)
    :param number_of_samples: how many samples to get from the directory, if none all the files in the folder will be considered.
    :return: a list of dataframes obtained from the csv files if number_of_samples is not None a dictionary is returned to trace back the original flight numbers from the list indices.
    '''
    csvs=[]
    flight_number = 0
    name_list = alpha_numeric_sort(os.listdir(dirname))
    if flight:
        try:
            name_list = [name for name in name_list if flight == name.replace(".", "_").split("_")[-2]]
            assert len(name_list) == 1
        except:
            raise Exception("The flight number given does not seem to be in the folder")
    elif number_of_samples:
        name_list = sample(name_list, number_of_samples)
    try:
        flight_number_dict = {index: int(name.replace(".", "_").split("_")[-2])for index, name in enumerate(name_list)}
    except:
        raise Exception(f"The last part of the name should be a number")
    for idx, csv_file in enumerate(name_list):
        if not (idx in excl_index):
            df = pd.read_csv(f"{dirname}/{csv_file}")[naming_convention.keys()]
            df = df[df['time'].notna()]
            df.sort_values(by='time', inplace = True)
            df.reset_index(inplace=True, drop=True)
            df.rename(columns=naming_convention, inplace=True)
            plt.show()
            if flight_numbering:
                df['n'] = flight_number
                flight_number += 1
            csvs.append(df)
    return csvs, flight_number_dict

def fill_time_gaps(df: pd.DataFrame, time_col:str ='ts'):
    '''
    Fills the time gaps with NaN values in the other columns in a dataframe

    :param df: the dataframe in which to fill the time gaps
    :param time_col: the label of the column that contains the time information.
    :return: a new dataframe with no time gaps and Nan values for the remaining columns where there used to be time gaps.
    '''

    new_df = pd.DataFrame(np.nan, index=range(int(df[time_col].min()), int(df[time_col].max())), columns=df.columns)
    new_df = new_df.combine_first(df.set_index(time_col))
    new_df[time_col] = new_df.index
    return new_df

def nan_same_values(df:pd.DataFrame):

    '''
    Replaces repeated values with NaNs.
    (Repeated values indicate fictional data  as in absence of new values old values are repeated.)

    :param df: dataframe on which repeated values should be removed
    :return: a dataframe where repeated values are removed, and amount of repeated values.
    '''

    count = df.count()
    df = df.where(cond= (df.diff() != 0) | (df == 0), other=np.nan)
    count -= df.count()
    return df, count.sum()

def initialize_quality_df(overview_df:pd.DataFrame, dfs_list:List[pd.DataFrame], number_conversion:dict) -> None:
    '''
    :param overview_df: Dataframe containing information of all flights
    :param dfs_list: list of Dataframes containing each flight
    :param number_conversion: dictionary that maps the indexes to the original flight number
    :return:
    '''
    overview_df['flight_number'] = overview_df.index
    quality_df = overview_df.loc[number_conversion.values()]
    quality_df['flight_number'] = quality_df.index
    quality_df.reset_index(inplace=True, drop=True)
    quality_df['number_of_datapoints'] = [(df['ts'].iloc[-1] - df['ts'].iloc[0]) * df.shape[1] for df in dfs_list]
    quality_df['total_missing_values'] = [(df['ts'].iloc[-1] - df['ts'].iloc[0]) * df.shape[1] - df.count().sum() for df in dfs_list]
    return quality_df


### Outlier removal ###

def apply_cut_off(cut_off_points:dict, df:pd.DataFrame, i:int, print_warning:bool=False) -> (pd.DataFrame, float):
    '''
    Removes outliers based on the amount of change from one to the next.

    :param cut_off_points: a dictionary with the cut off points for maximum amount of allowed change in 1 time step e.g. {column: cut off value, ...}
    :param df: the dataframe to which to apply the cut off points
    :param i: a number to identify the dataframe (only necessary if warning is printed)
    :param print_warning: whether to print a warning for datarframes that have excessive amount of points cut off.
    :return:
    '''
    diff_foward = df.diff().abs()
    diff_backward = df.diff(-1).abs()
    count = df.count()
    for col, val in cut_off_points.items():
        cut_off_mask = ((abs(diff_foward[col]) > val) & (abs(diff_backward[col]) > val))
        df.loc[cut_off_mask, col] = np.nan
    count -= df.count()
    if print_warning and any(count > 100):
        print(f"Warning: amount of cut offs for {i}:\n{count}\n")
    return df, count.sum()


def outlier_extra_alt(df:pd.DataFrame, cut_off_points:dict, window_len:int =M_WDW, smoothing_window_len:int =S_WDW, max_nan:int=None):
    '''
    An additional outlier removal that can be used after besides the apply_cut_off.
    It compares the distance from a point to the median of around this point (wind_len) and applies a median filter on the data (smoothing_window_len).

    :param df: The dataframe of which outliers have to be removed
    :param cut_off_points: The amount of change allowed from median to point.
    :param window_len: The window size for the points taken into account to compute the median around a point.
    :param smoothing_window_len: The window size for smoothing after the outlier removal. None indicates no smoothing
    :return:
    '''
    cols = cut_off_points.items()
    glob_count = 0
    wind = window_len
    for col, val in cols:
        if not max_nan is None:
            wind = min(max_nan[col]+1, window_len)
            if (wind % 2) == 0:
                wind += 1
        median_col = median_filter(df[col], window_len=wind)
        count = df.count()
        outlier_mask = abs(df[col] - median_col) > val
        df.loc[outlier_mask, col] = np.nan
        if not col == 'alt_backup':
            glob_count += (count - df.count()).sum()
        if smoothing_window_len:
            temp_count = df.count()
            outlier_mask_wdw = abs(df[col].diff(2 * smoothing_window_len)) > (2 * smoothing_window_len * val)
            df.loc[outlier_mask_wdw, col] = np.nan
            temp_count -= df.count()
            if not col == 'alt_backup':
                glob_count += temp_count.sum()
            df[col] = median_filter(df[col], window_len=smoothing_window_len)
    return df, glob_count

def cluster_outlier(df:pd.DataFrame, col:str='alt', max_dist=C_DIST, cut_off = C_CUT):
    '''
    Finding outliers using DBSCAN

    :param df:
    :param col:
    :param max_dist:
    :param cut_off:
    :return:
    '''
    no_na_df = df[[col, 'ts']].dropna()
    X = no_na_df.to_numpy()
    if len(X) < 1:
        return df, 0
    cluster = DBSCAN(eps=max_dist, min_samples=1).fit(X)
    no_na_df['labels'] = cluster.labels_
    counts = no_na_df['labels'].value_counts(ascending=True)
    no_na_df['counts'] = [counts[lab] for lab in no_na_df['labels']]
    count = df[col].count()
    df.loc[no_na_df[no_na_df['counts'] <= cut_off].index, col] = np.nan
    count -= df[col].count()
    return df, count

def alt_backup_fill(df, dist = F_TH):
        fill_alts = df['alt'].count()
        filled = 0
        if abs((df['alt'] - df['alt_backup']).median()) < dist:
            df.loc[df['alt'].isna(), 'alt'] = df['alt_backup']
            filled = 1
        elif df['alt_backup'].count() > df['alt'].count():
            df.loc[:, 'alt'] = df.loc[:, 'alt_backup']
            filled = 2
        fill_alts = df['alt'].count() - fill_alts
        return df, fill_alts, filled

### Filtering ###

def get_windowed_slice(x, idx, half_wind):
    '''
    Gets a window slice of signal x.

    :param x: the signal
    :param idx: index at which the center of the window should be
    :param half_wind: half the window size ((odd_window - 1)/2
    :return: the slice of signal x with size 2*half_wind + 1 and center idx.
    '''
    if idx < half_wind or idx > (len(x) - half_wind):
        median_pad(x, half_wind)
    return x[idx-half_wind:idx+half_wind]

def median_pad(x, half_wind):
    '''
    Pads a signal with the median of the first/last half_wind points.

    :param x: one dimensional list or array to be padded
    :param half_wind: half a window size according to the filter window size
    :return: a one dimensional array padded with the median of the first and last window.
    '''
    return np.concatenate([np.full(half_wind, np.nanmedian(x[:(2*half_wind+1)])), x, np.full(half_wind, np.nanmedian(x[-(2*half_wind+1):]))])


def median_filter(x, window_len= M_WDW):
    '''
    Applies a median filter that can handle Nans.

    :param x: one dimensional list or array to apply the filter to
    :param window_len: the window size for filtering
    :return: a filtered numpy array of the same size as x
    '''
    x = np.array(x)
    if window_len % 2 == 0:
        raise ValueError("Median filter can only handle odd window sizes for now.")
    if x.ndim != 1:
        raise ValueError("Median filter only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError(f"Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    half_wind = int((window_len-1)/2)
    pad_x = np.pad(x, (half_wind, half_wind), 'minimum')
    # pad_x = median_pad(x, half_wind)
    y = [np.nanmedian(get_windowed_slice(pad_x, idx, half_wind)) for idx in range(half_wind, half_wind+len(x))]
    return y


### Results ###

def get_comparison(old_dfs, new_dfs, columns):
    '''
    Compare two lists of dataframes of same size and order

    :param old_dfs: The old list of dataframe
    :param new_dfs: The new list of dataframes which is the same as old_dfs except for the values.
    :param columns: the columns that wish to be compared
    :return: a list of dataframes where each dataframe has the old and new columns next to each other.
    '''
    comparative_dfs = deepcopy(new_dfs)
    no_smooth_cols = set(new_dfs[0].columns) - set(columns)
    for idx, df in enumerate(comparative_dfs):
        df.drop(columns=no_smooth_cols, inplace=True)
        df[[(col+'_old') for col in columns]] = old_dfs[idx][columns]
        df.sort_index(axis=1, inplace=True)
    return comparative_dfs

def plot_comparisons(comparative_df, columns, title=None, save=None, alt_backup=None, quality_statement=None):
    '''
    Show a plot that compares the old and new signal for each of the columns based of comparative_df

    :param comparative_df: a dataframe created with the "get_comparison" function.
    :param columns: the columns for which comparison plots should be shown
    :param number: the number of flight if wanted in the title
    :param save: the path including the file name where to save the files to.
    :return:
    '''
    color_dict = {'good': 'tab:green', 'average': 'tab:orange', 'bad': 'tab:red'}
    n = len(columns)
    if 'ts' in columns:
        n -= 1
    fig, ax = plt.subplots(n, figsize=(10, 20))
    ax[0].scatter(x=(comparative_df["ts"] - comparative_df["ts"].min()), y=(comparative_df["alt_old"] * 3.28984), c="r",
                  marker='.', alpha=0.4,
                  label="original")
    ax[0].scatter(x=(comparative_df["ts"]-comparative_df["ts"].min()), y=(comparative_df["alt"]*3.28984), c="b", marker='.',
                  alpha=0.1, label="processed")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Altitude (ft)")
    ax[0].legend()

    ax[1].scatter(x=(comparative_df["ts"] - comparative_df["ts"].min()), y=comparative_df["spd_old"] * 1.94384, c="r",
                       marker='.', alpha=0.4,
                       label="original")
    ax[1].scatter(x=(comparative_df["ts"] - comparative_df["ts"].min()), y=comparative_df["spd"] * 1.94384, c="b",
                       marker='.', alpha=0.1, label="processed")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Speed (kts)")
    ax[1].legend()

    if title:
        if quality_statement:
            t = fig.suptitle(title, y=0.9)
            plt.setp(t, color=color_dict[quality_statement], size=14)
        else:
            fig.suptitle(title, y=0.9)
    if save:
        plt.savefig(f"{save}.png")
        plt.close(fig)
        plt.clf()
        plt.cla()
    else:
        plt.show()


def consecutive_nans(df):
    '''
    Put amount of consecutive Nans in a list for each column in the dataframe.

    :param df: The dataframe on which to count consecutive Nans.
    :return: a dictionary where the key is the name of the column and the value is a list of consecutive Nans
    '''
    dict = {}
    for col in df.columns:
        nans = df[col].isnull().astype(int).groupby(df[col].notnull().astype(int).cumsum()).sum()
        dict[col] = nans
    dict = pd.DataFrame(dict)
    return dict

def compute_changes(new_df, old_df, fill_event, cut_off_points):
    if 'alt_backup' in cut_off_points.keys():
        cut_off_points.pop('alt_backup')
        new_df.drop(columns=['alt_backup'], inplace=True)
        if fill_event == 2: #alt has been replaced with alt_backup
            old_df.loc['alt'] = old_df['alt_backup']
        old_df.drop(columns=['alt_backup'], inplace=True)
    diffs = old_df.subtract(new_df)
    s = 0
    for col, val in cut_off_points.items():
        cut_off_mask = ~(abs(diffs[col]) <= val)
        s += cut_off_mask.sum(axis=0)
    return s


def save_quality_dfs(quality_df, folder='.'):

    # Route
    route_quality = quality_df.groupby(by=['departure', 'destination'])
    route_quality_df = route_quality.mean()
    route_quality_df.columns = [str(col) + ' (mean over flights)' for col in route_quality_df.columns]
    route_quality_df['number_of_flights'] = route_quality.size()
    route_quality_df.to_csv(f"{folder}/route_data_quality.csv")

    # One airport
    deps = quality_df.drop(columns=['destination', 'flight_number']).rename(columns={'departure': 'airport'})
    dests = quality_df.drop(columns=['departure', 'flight_number']).rename(columns={'destination': 'airport'})
    airport_quality_single = pd.concat([deps, dests]).groupby(by=['airport'])
    airport_quality_single_df = airport_quality_single.mean()
    airport_quality_single_df.columns = [str(col) + ' (mean over flights)' for col in airport_quality_single_df.columns]
    airport_quality_single_df['number_of_flights'] = airport_quality_single.size()
    airport_quality_single_df.to_csv(f"{folder}/one_aiport_data_quality.csv")

    # Pair of airports
    sets = quality_df.copy()
    sets['airport'] = sets.apply(lambda row: '-'.join(alpha_numeric_sort([str(row['departure']), str(row['destination'])])), axis=1)
    sets = sets.drop(columns=['departure','destination', 'flight_number'])
    airport_quality_pair = sets.groupby(by=['airport'])
    airport_quality_pair_df = airport_quality_pair.mean()
    airport_quality_pair_df.columns = [str(col) + ' (mean over flights)' for col in airport_quality_pair_df.columns]
    airport_quality_pair_df['number_of_flights'] = airport_quality_pair.size()
    airport_quality_pair_df.to_csv(f"{folder}/pair_aiport_data_quality.csv")

    quality_df.to_csv(f"{folder}/flight_data_quality.csv")

if __name__ == '__main__':

    while "src" in os.getcwd():
        os.chdir("..")
        print(f"Changed directory to {os.getcwd()}")

    print(os.listdir("data"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="The path to the folder where the csvs of the flights are stored.")
    parser.add_argument("--overview_file", type=str, default=None, help="The path to the overview file. If not provided quality statement will not be made.")
    parser.add_argument("--figs", type=bool, default=True, help="Whether to save the comparison figures")
    parser.add_argument("--save_folder", type=str, default=None, help="The path to the folder where the results and smoothed csvs should be stored.")
    parser.add_argument("--plot_columns", nargs="*", type=str, default=['ts', 'alt', 'spd', 'roc'],
                        help="The labels of the columns that should be plotted if figs=True")
    parser.add_argument("--flights", type=int, default=None, help="The amount of flights to pre process from the folder (they are randomly sampled from the folder). If None all flights will be taken into consideration.")
    parser.add_argument("--columns", nargs="*", type=str, default=['ts', 'alt', 'spd', 'roc'], help="The labels of the columns that need to be prprocessed.")
    parser.add_argument("--extra_outlier", nargs="*", type=str, default=['alt', 'spd', 'roc'], help="The labels of the columns on which additional outlier removal should be performed.")

    args = parser.parse_args()
    folder = args.folder
    save_folder = args.save_folder
    figs = args.figs
    number_of_flights = args.flights
    if number_of_flights == -1:
        number_of_flights = None
    overview_file = args.overview_file

    # Temporary changes
    smooth_cols = args.columns
    extra_outlier = args.extra_outlier

    plot_cols = args.plot_columns
    if not save_folder:
        save_folder = folder + '_preprocessed'

    if not os.path.exists(f'{save_folder}'):
        os.makedirs(f'{save_folder}')

    ### Initialization

    csvs, number_conversion = get_df_list(dirname=folder, number_of_samples=number_of_flights)
    print("Successfully imported files")
    dfs = [csv[smooth_cols].copy() for csv in csvs]
    dfs, rep_vals = zip(*[nan_same_values(df.copy()) for df in dfs])
    max_nans = [consecutive_nans(df.copy()).max() for df in dfs]
    diffs = [df.copy().diff() for df in dfs]
    diff_df = pd.concat(diffs)
    if overview_file:
        overview = pd.read_csv(overview_file)
        quality_df = initialize_quality_df(overview, dfs, number_conversion)
        quality_df['total_repeated_values'] = rep_vals
        quality_df['max_length_of_gap'] = [consecutive_nans(df).max().max() for df in dfs]


    ### Removing outliers

    cut_off_points = {'ts': 1, 'alt': 150, 'spd': 1.5, 'roc': 1.5}  # , 'alt_backup':150 }

    print(f"The outlier cut off points are:\n{cut_off_points}\n")

    dfs, outliers = zip(*[apply_cut_off(cut_off_points, df.copy(), idx) for idx, df in enumerate(dfs)])
    print(f"Found {sum(outliers)} outliers with cut off.")

    if overview_file:
        quality_df['total_outliers'] = outliers

    ### Filling time gaps & eliminating fictional data

        quality_df['number_of_time_gaps'] = [len(diff[diff['ts'] > 1]) for diff in diffs]

    dfs = [fill_time_gaps(df[smooth_cols].copy()) for df in dfs]

    if overview_file:
        for i, df in enumerate(dfs):
            df_time = math.ceil((df['ts'].iloc[-1] - df['ts'].iloc[0]) / 60)
            flight_number = number_conversion[i]
            overview_time = overview['duration'][flight_number]
            if abs(overview_time - df_time) > 1:
                print(f"Duration from overview does not correspond to csv for flight{flight_number}\n"
                      f"Overview time is {overview_time}, csv time is {df_time}")


    ### Extra outlier removal

    extra_cut_off = {col: cut_off_points[col] for col in extra_outlier}
    dfs, extra_outliers = zip(*[outlier_extra_alt(df.copy(), extra_cut_off, max_nan=max_nans[i]) for i, df in enumerate(dfs)])
    if overview_file:
        quality_df['total_outliers'] += extra_outliers
    print(f"Found {sum(extra_outliers)} outliers with extra altitude outlier.")

    ### Cluster outlier removal
    dfs, cluster_outliers = zip(*[cluster_outlier(df.copy(), col='alt') for i, df in enumerate(dfs)])
    if overview_file:
        quality_df['total_outliers'] += cluster_outliers
    print(f"Found {sum(cluster_outliers)} outliers with cluster altitude outlier.")


    ###  Interpolation

    dfs = [df.interpolate(method='linear', limit_area='inside', axis=0) for df in dfs]

    ### Results

    time_index_csvs = [df.set_index('ts', drop=False) for df in csvs]

    if overview_file:
        quality_df['invalid_values'] = [compute_changes(dfs[i].copy(), time_index_csvs[i].copy(), 0, cut_off_points)
                                        for i in range(len(dfs))]

        quality_df['percentage_problematic_values'] = quality_df[
            ['total_outliers', 'total_missing_values', 'total_repeated_values']].sum(axis=1)
        quality_df['percentage_problematic_values'] = quality_df['percentage_problematic_values'] / quality_df[
            'number_of_datapoints'] * 100
        quality_df['percentage_invalid_values'] = quality_df['invalid_values'] / quality_df['number_of_datapoints'] * 100

        bad = quality_df['percentage_invalid_values'].quantile(0.9)
        good = quality_df['percentage_invalid_values'].quantile(0.1)
        statement = []
        for i, r in quality_df.iterrows():
            # print(r)
            if pd.isna(r['departure']) or pd.isna(r['destination']):
                statement.append('bad')
            elif r['percentage_invalid_values'] <= good:
                statement.append('good')
            elif r['percentage_invalid_values'] >= bad:
                statement.append('bad')
            else:
                statement.append('average')
        quality_df['statement'] = statement

        if not os.path.exists(f'{save_folder}/reports'):
            os.makedirs(f'{save_folder}/reports')
        save_quality_dfs(quality_df, folder=f"{save_folder}/reports")

    else:
        statement = [None for i in range(len(dfs))]


    if figs:
        if not os.path.exists(f'{save_folder}/images'):
            os.makedirs(f'{save_folder}/images')
        compare_dfs = get_comparison(time_index_csvs, dfs, ["ts", "alt", "spd"])
        for idx, df in enumerate(compare_dfs):
            title = f"Flight number {idx} ({number_conversion[idx]})"
            plot_comparisons(df, ['alt', 'spd'], title=title,
                     save=f"{save_folder}/images/flight_{number_conversion[idx]}", quality_statement=statement[idx])

    if not os.path.exists(f'{save_folder}/csvs'):
        os.makedirs(f'{save_folder}/csvs')

    for i, df in enumerate(dfs):
        df.reset_index(inplace=True)
        i1 = df[['alt', 'spd', 'roc']].apply(pd.Series.first_valid_index).max()
        i2 = df[['alt', 'spd', 'roc']].apply(pd.Series.last_valid_index).min()
        df = df.loc[i1:i2]
        df.reset_index(inplace=True)
        df.loc[:, 'roc'] = df['roc'] * 196.85  # m/s -> ft/min
        df.loc[:, 'spd'] = df['spd'] * 1.94384  # m/s -> kt
        df.loc[:, 'alt'] = df['alt'] * 3.28084  # m -> ft
        df.to_csv(f"{save_folder}/csvs/flight_{number_conversion[i]}.csv", index=False)

        df.ts = df.ts - df.ts[0]