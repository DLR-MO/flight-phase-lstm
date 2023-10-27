# SPDX-FileCopyrightText: 2023 German Aerospace Center
#
# SPDX-License-Identifier: MIT

'''
Author: Alexander Kamtsiuris (alexander.kamtsiuris@dlr.de)

Flight phase labeling of X-plane recorded data
'''

import argparse
import os
import re

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from flight_phase_finder_core import find_flight_phases
from matplotlib.lines import Line2D
import shutil
import random
import traceback

def alpha_numeric_sort(list):
    '''Sort the given iterable in alphanumeric order.'''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list, key=alphanum_key)

def gear_lever_select_down(gear1, gear2, gear3):
    gear = np.nanmean(np.vstack((gear1, gear2, gear3)),axis=0)

    gear_diff = np.hstack((np.diff(gear), 0.))
    
    idx = np.where((gear.astype(int) == 1) | (gear_diff > 0.))
    
    param = np.zeros(gear.shape).astype(int)
    param[idx] = 1
    
    return param

def get_nose_gear_steering_angle(angles,onground):
    
    idx = np.where(~(onground.astype(int) == 1))
    
    angles[idx] = 0.
    
    return angles

def get_epr_take_off_channel(epr):
    param = np.full(epr.shape,np.max(epr))
    return param

def get_epr_command_channel(epr,lever_pos):
    param = 1. + lever_pos * (np.max(epr) - 1.)
    return param

def get_datetime(hobbs_time):
    t0 = 1602241094.
    hour_to_seconds = 3600.
    
    time = t0 + hobbs_time * hour_to_seconds
    
    param = pd.to_datetime(time, unit="s")
    
    return param

def get_gear_compressed(norm_force):

    param = np.zeros(norm_force.shape)

    idx = np.where(norm_force.astype(int) > 0)
    param[idx] = 1
    return param

def get_altitude_rate(altitude,hobbs_time):
    hour_to_seconds = 3600.
    ft_s_to_ft_min = 60.
    dt = np.diff(hobbs_time * hour_to_seconds)
    dalt = np.diff(altitude)
    
    rate = dalt/dt * ft_s_to_ft_min
    
    rate = np.hstack((rate, rate[-1]))
    return rate


m_s_to_ft_min = 196.8504
xplane_rate = 10
adsb_rate = 1
conversion_rate = int(xplane_rate / adsb_rate)
naming_convention = {"DATE_TIME": 'ts',
                     "ALTITUDE_PRESSURE": 'alt',
                     "GROUND_SPEED": 'spd',
                     'ALTITUDE_RATE': 'roc',
                     "FLIGHT_PHASES_DETECTED": 'phase',
                     "LATITUDE": "lat",
                     "LONGITUDE": 'lon'}

def process_file(xplane_df):

    main_landing_gear_compressed = get_gear_compressed(xplane_df['   norm_,___lb '].values)

    df = pd.DataFrame({
        "DATE_TIME": get_datetime(xplane_df['   hobbs,_time ']),
        "ALTITUDE_PRESSURE": xplane_df["   __alt,ftmsl "],
        "ALTITUDE_RATE": xplane_df["   __VVI,__fpm "],
        "GROUND_SPEED": xplane_df["   Vtrue,_ktgs "],
        "CMD_EPR_1": get_epr_command_channel(xplane_df["   EPR_1,_part "].values,
                                                         xplane_df["   thro1,_part "].values),
        "TGT_EPR_1": get_epr_take_off_channel(xplane_df["   EPR_1,_part "].values),
        "CMD_EPR_2": get_epr_command_channel(xplane_df["   EPR_2,_part "].values,
                                                            xplane_df["   thro2,_part "].values),
        "TGT_EPR_2": get_epr_take_off_channel(xplane_df["   EPR_2,_part "].values),
        "GEAR_LVR_DOWN": gear_lever_select_down(xplane_df["   _gear,__rat "].values,
                                                         xplane_df["   _gear,__rat .1"].values,
                                                         xplane_df["   _gear,__rat .2"].values),
        "GEAR_COMPR": main_landing_gear_compressed,
        "STEER_ANG": get_nose_gear_steering_angle(xplane_df["   _gear,__deg "].values,
                                                                 main_landing_gear_compressed),
        "LATITUDE": xplane_df["   __lat,__deg "] - xplane_df['   __lat,orign '],
        "LONGITUDE": xplane_df["   __lon,__deg "] - xplane_df['   __lon,orign ']
    })
    return df


colormap = {
        "taxi": "black",
        "take-off": "pink",
        "initial climb": "yellow",
        "climb": "green",
        "cruise": "blue",
        "descent": "orange",
        "approach": "brown",
        "landing": "cyan",
    }

number_to_label = {
    0: "taxi",
    1: "take-off",
    2: "initial climb",
    3: "climb",
    4: "cruise",
    5: "descent",
    6: "approach",
    7: "landing"
}

legend_lines = []
for lab, col in colormap.items():
    legend_lines.append(Line2D([0], [0], color=col, label=lab))


def show_ground_truth(labels, df, save=False, number=None):
    colors = [colormap[number_to_label[l]] for l in labels]

    plt.scatter(df['ts'], df['alt'], marker=".", c=colors, lw=0)
    plt.ylabel("altitude (ft)")
    plt.xlabel("time (s)")

    plt.legend(handles=legend_lines, prop={'size': 8})

    if save:
        plt.savefig(f"{save_folder}/figures/flight_{number}.png")

        plt.clf()
    else:
        plt.draw()
        plt.waitforbuttonpress(-1)
        fig = plt.gcf()
        return fig

if __name__ == '__main__':

    '''
    Transform X-plane txt to trajectory csvs and store their labels
    '''
    while "src" in os.getcwd():
        os.chdir("..")
        print(f"Changed directory to {os.getcwd()}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default='data/xplane_raw', # xplane_raw
                        help="The path to the folder where the csvs of the flights are stored.")
    parser.add_argument("--save_folder", type=str, default='data/temp', # preprocessed
                        help="The path to the folder where the trajectory data, plots and labels should be saved")

    args = parser.parse_args()
    folder = args.folder
    save_folder = args.save_folder

    if not os.path.exists(f'{save_folder}'):
        os.makedirs(f'{save_folder}')
    if not os.path.exists(f'{save_folder}/trajectories'):
        os.makedirs(f'{save_folder}/trajectories')
    if not os.path.exists(f'{save_folder}/labels'):
        os.makedirs(f'{save_folder}/labels')
    if not os.path.exists(f'{save_folder}/figures/discarded'):
        os.makedirs(f'{save_folder}/figures/discarded')

    flights = []
    flight_times = []
    ## Read files and make all columns numeric
    for idx, file in enumerate(alpha_numeric_sort(os.listdir(folder))):
        print(f"\n file {file}", end=' ')
        if "'../raw_data/xplane" == folder:
            min_time = 12000
            xplane = pd.read_csv(f"{folder}/{file}", delimiter=";", header=2).dropna()
        else:
            min_time = 1200
            xplane = pd.read_csv(f"{folder}/{file}", delimiter="|").dropna()
            # print(xplane.columns)
            for i, type in enumerate(xplane.dtypes):
                if type != np.float64:
                    col = xplane.columns[i]
                    try:
                        xplane.loc[:, col] = xplane[col].apply(pd.to_numeric)
                    except:
                        print(f"Could not convert column {col} ({i}) from {type} to float")
                        pass

        ## A single flight is identified by the period where the engine is on if this period is above a threshold
        try:
            engine = np.multiply(xplane['   rpm_1,engin '] == 0, 1)
            engine_on = np.where(np.diff(engine) == -1)[0]
            engine_off = np.where(np.diff(engine) == 1)[0]
            if engine_off[0] < engine_on[0]:
                engine_off = engine_off[1:]
            if engine_on[-1] > engine_off[-1]:
                engine_on = engine_on[:-1]
            assert(len(engine_on) == len(engine_off))
            for i in range(len(engine_on)):
                if engine_off[i] - engine_on[i] > min_time:
                    f = xplane[engine_on[i]:engine_off[i]+1].copy().reset_index()
                    time_tuple = (f['   hobbs,_time '].iloc[0], f['   hobbs,_time '].iloc[-1])
                    if not time_tuple in flight_times:
                        flight_times.append(time_tuple)
                        f['   hobbs,_time '] -= xplane['   hobbs,_time '][engine_on[i]-1]
                        flights.append(process_file(f))
                        print(len(flights)-1, end=' ')

        except Exception as e:
            print(f"Could not process file {file}, {e}")


    # Run flight phase finder
    failed = 0
    print(f"Found {len(flights)} flights.")
    # unks = np.zeros(len(flights))
    min_alt = 7500
    for i, df in enumerate(flights):
        try:
            print("Flight ", i)
            if (df["DATE_TIME"].iloc[-1] - df["DATE_TIME"].iloc[0]).total_seconds() < 3600 and df['ALTITUDE_PRESSURE'].max() > 36000:
                raise Exception("Short flight with abnormal altitude.")
            if df['ALTITUDE_PRESSURE'].max() < min_alt:
                raise Exception(f"Flight does not reach {min_alt} ft.")
            df_full = find_flight_phases(df)
            df_full.rename(columns=naming_convention, inplace=True)


            start_taxi = np.where(df_full["phase"] == 0)[0][0]
            end_taxi = np.where(df_full["phase"] == 0)[0][-1]
            df_sampled = df_full.loc[start_taxi:end_taxi, :]
            df_sampled.loc[:, 'ts'] = [np.floor(pd.Timedelta(t-df_sampled["ts"].values[0]).total_seconds()) for t in df_sampled["ts"].values]
            mask = df_sampled['ts'].diff() > 0
            df_sampled = df_sampled[mask]

            df_trajectory = df_sampled[naming_convention.values()]
            df_trajectory = df_trajectory.drop(columns='phase')
            df_trajectory.to_csv(f"{save_folder}/trajectories/flight_{i}.csv", index=False)
            df_labels = df_sampled["phase"]
            df_labels.to_csv(f"{save_folder}/labels/flight_{i}.csv", index=False)

            show_ground_truth(labels=df_sampled['phase'], df=df_sampled, save=True, number=i)

        except Exception as e:
            print(f"Could not process the flight in figure. {e}")
            print(traceback.format_exc())
            fig, ax1 = plt.subplots()
            ax1.plot(df['ALTITUDE_PRESSURE'])
            if "FLIGHT_PHASES_DETECTED" in df.columns:
                ax2 = ax1.twinx()
                ax2.plot(df["FLIGHT_PHASES_DETECTED"], color='gray')
            plt.title(f"{e}")
            fig.savefig(f"{save_folder}/figures/discarded/flight_{i}.png")
            plt.clf()
            failed += 1
            pass
    print(f"Successfully processed {len(flights) - failed} out of {len(flights)} flights.")

    print("Splitting files in training, test and validation")

    files = os.listdir(f"{save_folder}/trajectories/")
    random.shuffle(files)

    files_train_index = int(len(files)*0.85)
    print(f"{files_train_index} files in train, "
          f"{len(files) - files_train_index} files in test.")

    if not os.path.exists(f"{save_folder}/trajectories_train/"):
        os.mkdir(f"{save_folder}/trajectories_train/")
        os.mkdir(f"{save_folder}/labels_train/")
    for file in files[:files_train_index]:
        shutil.copy(f"{save_folder}/trajectories/{file}", f"{save_folder}/trajectories_train")
        shutil.copy(f"{save_folder}/labels/{file}", f"{save_folder}/labels_train")

    if not os.path.exists(f"{save_folder}/trajectories_test/"):
        os.mkdir(f"{save_folder}/trajectories_test/")
        os.mkdir(f"{save_folder}/labels_test")
    for file in files[files_train_index:]:
        shutil.copy(f"{save_folder}/trajectories/{file}", f"{save_folder}/trajectories_test")
        shutil.copy(f"{save_folder}/labels/{file}", f"{save_folder}/labels_test")
