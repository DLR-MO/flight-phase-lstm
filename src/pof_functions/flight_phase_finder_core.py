'''
Author: Alexander Kamtsiuris (alexander.kamtsiuris@dlr.de)

Implementation of ICAO ADREP event phases of A320 FDR data
(https://www.icao.int/safety/airnavigation/AIG/Documents/ADREP%20Taxonomy/ECCAIRS%20Aviation%201.3.0.12%20(VL%20for%20AttrID%20%20391%20-%20Event%20Phases).pdf)
'''

import numpy as np
import pandas as pd
from datetime import timedelta

def find_zero_runs(a):
    '''Create an array that is 1 where a is nonzero, and pad each end with an extra 0.'''
    iszero = np.concatenate(([0], (np.asarray(a) == 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def find_cruise_levels(data):
    climb_rate_limit = 500. # ft/min
    min_cruise_len = 15  # sec
    cruise_alt = data["ALTITUDE_PRESSURE"].max() * 0.9

    data["ALTITUDE_ABOVE_DEST_AIRPORT"] = data["ALTITUDE_PRESSURE"] - data.loc[data.index[(data["TAXI_FLAG"]) & (data.index > data.index[int(len(data) / 2)])].min(), "ALTITUDE_PRESSURE"]
    # put zero where climb rate is below the limit
    data["ALTITUDE_RATE_ZEROED"] = data["ALTITUDE_RATE"]
    data.loc[data["ALTITUDE_RATE"].abs() < climb_rate_limit, "ALTITUDE_RATE_ZEROED"] = 0.
    data["CRUISE_FLAG"] = False
    zero_indices = find_zero_runs(data["ALTITUDE_RATE_ZEROED"].values)
    # cruise_alt = data["ALTITUDE_PRESSURE"].max() - cruise_alt_allowance
    zero_indices_mean_alt = [data[elem[0]:elem[-1]]["ALTITUDE_PRESSURE"].mean() for elem in zero_indices]

    #iterate through zero runs, analyse and apply cruise phase number
    for i, elem in enumerate(zero_indices):
        #if len of phase below limit, dont count as cruise phase
        if elem[1] - elem[0] >= min_cruise_len and zero_indices_mean_alt[i] >= cruise_alt:
            data.iloc[elem[0]:elem[1], data.columns.get_loc("CRUISE_FLAG")] = True

    return data

def link_cruise_phases(data):
    # between first and last cruise phase index, assign cruise flag is TRUE
    cruise_alt = data["ALTITUDE_PRESSURE"].max() * 0.9
    try:
        first_cruise_idx = np.where(data["CRUISE_FLAG"].values)[0][0]
        last_cruise_idx = np.where(data["CRUISE_FLAG"].values)[0][-1]
        data.iloc[first_cruise_idx:last_cruise_idx, data.columns.get_loc("CRUISE_FLAG")] = True
    except IndexError:
        raise Exception("Did not find cruise phase.")
    if (data.loc[data["CRUISE_FLAG"], "ALTITUDE_PRESSURE"] < cruise_alt).any():
        raise Exception("Descended too low during cruise.")
    return data

def find_cruise(data):
    data = find_cruise_levels(data)
    data = link_cruise_phases(data)
    return data

def find_taxi(data):

    limit_factor = 1
    
    #main landing gear compressed and commanded EPR of any engine not at Take Off EPR
    data["TAXI_FLAG"] = (data["GEAR_COMPR"].values == 1.) & ((data["CMD_EPR_1"] <= limit_factor * data["TGT_EPR_1"]) & (data["CMD_EPR_2"] <= limit_factor * data["TGT_EPR_2"]))
    first_taxi = data[data["TAXI_FLAG"]].index[0]
    last_taxi = data[data["TAXI_FLAG"]].index[-1]
    data_len_0 = len(data)
    data.drop(index=data.index[:first_taxi], inplace=True)
    data.drop(index=data.index[last_taxi+1:], inplace=True)
    data.reset_index(inplace=True)
    if len(data) < data_len_0:
        print("Trimmed flight to taxi phases")
    return data

def find_take_off(data):
    
    data["ALTITUDE_ABOVE_ORIGIN_AIRPORT"] = data["ALTITUDE_PRESSURE"] - data.loc[data.index[(data["TAXI_FLAG"]) & (data.index < data.index[int(len(data) / 2)])].max(), "ALTITUDE_PRESSURE"]

    limit_factor = 0.8
    #altitude above airport elevation equal or lower than 35 ft, gear down selected, EPR commanded similar to TAKE OFF EPR

    data["TAKE_OFF"] = (data["ALTITUDE_ABOVE_ORIGIN_AIRPORT"] <= 35.) & (data["GEAR_LVR_DOWN"] == 1.) & ((data["CMD_EPR_1"] >= limit_factor * data["TGT_EPR_1"])|(data["CMD_EPR_2"] >= limit_factor * data["TGT_EPR_2"]))

    first_taxi_end_idx = data[(data["TAXI_FLAG"]) & (data.index < data.index[int(len(data)/2)])].index.max()
    take_off_starts = np.where(data["TAKE_OFF"].diff(-1) > 0)[0]
    data["TAKE_OFF_FLAG"] = (data["TAKE_OFF"]) & (data.index >= data.index.values[take_off_starts[take_off_starts < first_taxi_end_idx].max()]) & (data.index <= data.index.values[first_taxi_end_idx])
    return data

def find_climb(data):
    #find zero runs in cruise phases
    #check if before zero run is take off
    data["CLIMB_FLAG"] = False
    zero_indices = find_zero_runs(data["FLIGHT_PHASES_DETECTED"].values)
    #extend with 0 at the end to avoid index error
    flight_phases_extended = np.hstack((data["FLIGHT_PHASES_DETECTED"].values, 0.))
    for i, elem in enumerate(zero_indices):
        idx1 = elem[0] - 1
        idx2 = elem[-1]
        if flight_phases_extended[idx1] == 2 and flight_phases_extended[idx2] == 5:
            data.iloc[elem[0]:elem[-1], data.columns.get_loc("CLIMB_FLAG")] = True
    return data

def find_top_of_descent():
    return

def find_descent(data):
    # find zero runs in cruise phases
    # check if before zero run is cruise
    # no cruise after descent initiation possible!!!!
    # everything between 5 and 2 phases is descent if it occurs after top of descent phase
    # find index for top of descent
    # phases with continuous negative climb rate
    descent_time_limit = 120 #s used to be 120
    descent_rate_limit = -10
    descent_positive_rate_allowed = 1000
    data["DESCENT_FLAG"] = False
    data["ALTITUDE_RATE_NEGATIVE"] = data["ALTITUDE_RATE"] < descent_rate_limit
    zero_indices = find_zero_runs(1 - data["ALTITUDE_RATE_NEGATIVE"].values)
    for i, elem in enumerate(zero_indices):
        phase_time = data["DATE_TIME"].iloc[elem[-1]] - data["DATE_TIME"].iloc[elem[0]]
        phase_time = phase_time.total_seconds()

        #phase time above threshold and cruise phase beforehand existing, 
        #top of climb is found
        if phase_time > descent_time_limit and (data["FLIGHT_PHASES_DETECTED"].values[:elem[0]] == 5.).any():
            descent_mask = (data.index > data.index.values[elem[0]]) & (data.index < data.index[(data["TAXI_FLAG"]) & (data.index > data.index[int(len(data) / 2)])].min())
            if not((data.loc[descent_mask, "ALTITUDE_RATE"] > descent_positive_rate_allowed).any()):
                data.loc[descent_mask, "DESCENT_FLAG"] = True
                break
    return data

def find_initial_climb(data):
    data["INITIAL_CLIMB_FLAG"] = (data["ALTITUDE_ABOVE_ORIGIN_AIRPORT"] <= 1000.) & (data["CLIMB_FLAG"])
    return data

def find_approach(data):
    data["APPROACH_FLAG"] = data["DESCENT_FLAG"] & (data["ALTITUDE_ABOVE_DEST_AIRPORT"] <= 1000.)
    return data

def find_landing(data):
    """
    FROM LANDING FLARE TO STOP or TO EXITING RUNWAY
    """
    nose_wheel_steering_angle_limit = 3. # deg
    flare_time = 5. # s (flare not detectable due to noise => average time from real flight data used)

    data.index > data[data["TAKE_OFF_FLAG"]].index.max()
    data["ZERO_GROUND_SPEED_FLAG"] = data["GROUND_SPEED"] <= 5.
    data["STEERING_FLAG"] = np.abs(data["STEER_ANG"].values) > nose_wheel_steering_angle_limit
    
    contact_to_ground = (data["GEAR_COMPR"].values.astype(int)) & (data.index > data[data["CRUISE_FLAG"]].index.max())
    idx = find_zero_runs(contact_to_ground)
    flare_time_point = data["DATE_TIME"].iloc[idx[0][-1]] - timedelta(seconds=flare_time)
    data["AFTER_FLARE_INITIATION_FLAG"] = data["DATE_TIME"] >= flare_time_point
    
    idx_flare_initiation = np.where(data["AFTER_FLARE_INITIATION_FLAG"].values)[0][0]
    idx_first_steering = np.where((data["AFTER_FLARE_INITIATION_FLAG"] & data["STEERING_FLAG"]).values)[0][0]
    idx_first_stop = np.where((data["AFTER_FLARE_INITIATION_FLAG"] & data["ZERO_GROUND_SPEED_FLAG"]).values)[0][0]

    first_steering = np.zeros(len(data), dtype=bool)
    first_steering[idx_flare_initiation:idx_first_steering] = True
    
    first_stop = np.zeros(len(data), dtype=bool)
    first_stop[idx_flare_initiation:idx_first_stop] = True
    
    data["LANDING_FLAG"] = first_steering & first_stop

    return data

def find_flight_phases(data):
    '''
    :param data: pandas Dataframe containing FDR data
    :return: Dataframe 'data' with added column for flight phases 'FLIGHT_PHASES_DETECTED'
    '''

    data["FLIGHT_PHASES_DETECTED"] = np.zeros(len(data))
    data = find_taxi(data)
    if data["TAXI_FLAG"].sum() == 0:
        raise Exception('Did not find taxi phase.')
    data.loc[data["TAXI_FLAG"], "FLIGHT_PHASES_DETECTED"] = 1
    data = find_take_off(data)
    if data["TAKE_OFF_FLAG"].sum() == 0:
        raise Exception('Did not find take-off phase.')
    data.loc[data["TAKE_OFF_FLAG"], "FLIGHT_PHASES_DETECTED"] = 2
    data = find_cruise(data)
    if data["CRUISE_FLAG"].sum() == 0:
        raise Exception('Did not find cruise phase.')
    data.loc[data["CRUISE_FLAG"], "FLIGHT_PHASES_DETECTED"] = 5
    data = find_climb(data)
    if data["CLIMB_FLAG"].sum() == 0:
        raise Exception('Did not find climb phase.')
    data.loc[data["CLIMB_FLAG"], "FLIGHT_PHASES_DETECTED"] = 4
    data = find_initial_climb(data)
    if data["INITIAL_CLIMB_FLAG"].sum() == 0:
        raise Exception('Did not find initial climb phase.')
    data.loc[data["INITIAL_CLIMB_FLAG"], "FLIGHT_PHASES_DETECTED"] = 3
    data = find_descent(data)
    data = find_approach(data)
    data = find_landing(data)
    if data["DESCENT_FLAG"].sum() == 0:
        raise Exception('Did not find descent phase.')
    if data["APPROACH_FLAG"].sum() == 0:
        raise Exception('Did not find approach phase.')
    if data["LANDING_FLAG"].sum() == 0:
        raise Exception('Did not find landing phase.')
    data.loc[data["DESCENT_FLAG"], "FLIGHT_PHASES_DETECTED"] = 6
    data.loc[data["APPROACH_FLAG"], "FLIGHT_PHASES_DETECTED"] = 7
    data.loc[data["LANDING_FLAG"], "FLIGHT_PHASES_DETECTED"] = 8


    # Removing the very few remaining null phases
    data = data.reset_index()
    phase_transitions = np.where(abs(data["FLIGHT_PHASES_DETECTED"].diff()) > 0)[0]
    for i, current in enumerate(phase_transitions):
        if data["FLIGHT_PHASES_DETECTED"].values[current] == 0:
                if i < len(phase_transitions) - 1 and \
                        data["FLIGHT_PHASES_DETECTED"].values[current-1] == data["FLIGHT_PHASES_DETECTED"].values[phase_transitions[i+1]]:
                    data.iloc[current:phase_transitions[i+1], data.columns.get_loc("FLIGHT_PHASES_DETECTED")] = data["FLIGHT_PHASES_DETECTED"][current-1]
                elif data["FLIGHT_PHASES_DETECTED"].values[current] == 0:
                    raise Exception('Could not identify flight phase everywhere.')
    if len(np.where(abs(data["FLIGHT_PHASES_DETECTED"].diff()) > 0)[0]) > 8:
        raise Exception('FLight shows abnormal flight pattern (too many phase transitions).')
    data.loc[:, "FLIGHT_PHASES_DETECTED"] = data["FLIGHT_PHASES_DETECTED"].apply(lambda x: x-1)
    return data
