import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.signal
import warnings
import cv2
import math
import yaml
from locate_body_parts import body_part_locations
warnings.simplefilter("ignore")


def extract_dlc_coordinates(dlc_config_file, video_path):
    '''  EXTRACT RAW COORDINATES FROM DLC TRACKING    '''
    # read the saved coordinates file
    coordinates_file = glob.glob(os.path.dirname(video_path) + '\\*.h5')[0]
    DLC_dataframe = pd.read_hdf(coordinates_file)
    # get the name of the network
    DLC_network = os.path.basename(coordinates_file)
    DLC_network = DLC_network[DLC_network.find('Deep'):-3]
    # get the body parts used
    opened_dlc_config_file = open(dlc_config_file)
    parsed_dlc_config_file = yaml.load(opened_dlc_config_file)
    body_parts = parsed_dlc_config_file['bodyparts']
    # plot body part positions over time in a coordinates dictionary
    coordinates = {}
    # For each body part, get out the coordinates
    for i, body_part in enumerate(body_parts):
        # initialize coordinates
        coordinates[body_part] = np.zeros((3, len(DLC_dataframe[DLC_network][body_part]['x'].values)))
        # extract coordinates from loaded h5 file
        for j, axis in enumerate(['x', 'y']):
            coordinates[body_part][j] = DLC_dataframe[DLC_network][body_part][axis].values
        coordinates[body_part][2] = DLC_dataframe[DLC_network][body_part]['likelihood'].values
    return coordinates



def median_filter_and_transform_coordinates(processing, v):
    '''     FILTER AND TRANSFORM COORDINATES FROM DLC       '''
    # get the parameters
    plot = processing.show_extracted_coordinates_plot
    filter_kernel = processing.median_filter_duration_in_frames
    max_confidence = processing.minimum_confidence_from_dlc
    max_error = processing.maximum_error_drom_dlc
    x_offset, y_offset = processing.offset[v][0], processing.offset[v][1]
    # get the body parts used
    opened_dlc_config_file = open(processing.dlc_config_file)
    parsed_dlc_config_file = yaml.load(opened_dlc_config_file)
    body_parts = parsed_dlc_config_file['bodyparts']
    # fisheye correct the coordinates
    if os.path.isfile(str(processing.inverted_fisheye_correction_file)):
        inverse_fisheye_maps = np.load(processing.inverted_fisheye_correction_file)
    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), processing.coordinates[body_parts[0]].shape[1]))
    # loop across body parts to remove points with low confidence and to median filter
    for bp, body_part in enumerate(body_parts):
        # loop across axes
        for i in range(2):
            # remove coordinates with low confidence
            processing.coordinates[body_part][i][processing.coordinates[body_part][2] < max_confidence] = np.nan
            # interpolate nan values
            processing.coordinates[body_part][i] = np.array(pd.Series(processing.coordinates[body_part][i]).interpolate())
            processing.coordinates[body_part][i] = np.array(pd.Series(processing.coordinates[body_part][i]).fillna(method='bfill'))
            processing.coordinates[body_part][i] = np.array(pd.Series(processing.coordinates[body_part][i]).fillna(method='ffill'))
            # median filter coordinates (replace nans with infinity first)
            processing.coordinates[body_part][i] = scipy.signal.medfilt(processing.coordinates[body_part][i], filter_kernel)
            # remove coordinates with low confidence
            processing.coordinates[body_part][i][processing.coordinates[body_part][2] < max_confidence] = np.nan
        # put all together
        all_body_parts[:, bp, :] = processing.coordinates[body_part][0:2]
    # Get the median position of body parts in all frames (unless many points are uncertain)
    median_positions = np.nanmedian(all_body_parts, axis=1)
    num_of_nans = np.sum(np.isnan(all_body_parts[0, :, :]), 0)
    no_median = num_of_nans > (len(body_parts)/2)
    # Set up plot, if applicable
    if plot:
        fig = plt.figure('DLC coordinates', figsize=(14, 7))
        ax = fig.add_subplot(111)
    # loop across body parts to transform points to CCB
    for bp, body_part in enumerate(processing.coordinates):
        # get distance from median position for all frames
        distance_from_median_position = np.sqrt(
            (processing.coordinates[body_part][0] - median_positions[0, :]) ** 2 + (processing.coordinates[body_part][1] - median_positions[1, :]) ** 2)
        # loop across axes
        for i in range(2):
            # remove coordinates far from rest of body parts
            processing.coordinates[body_part][i][distance_from_median_position > max_error] = np.nan
            # remove coordinates where many body parts are uncertain
            processing.coordinates[body_part][i][no_median] = np.nan
            # correct any negative coordinates
            processing.coordinates[body_part][i][(processing.coordinates[body_part][i] < 0)] = 0
        # get index of uncertain points
        nan_index = np.isnan(processing.coordinates[body_part][i])
        # apply inverted fisheye remapping if applicable
        if os.path.isfile(str(processing.inverted_fisheye_correction_file)):
            # initialize transformed points array
            transformed_points = np.zeros(processing.coordinates[body_part].shape)
            # loop across axes
            for i in range(2):
                # convert original coordinates to registered coordinates
                transformed_points[i] = inverse_fisheye_maps[processing.coordinates[body_part][1].astype(np.uint16) + y_offset,
                                                             processing.coordinates[body_part][0].astype(np.uint16) + x_offset, i] \
                                                               - (x_offset*(1-i) + y_offset*(i))
        else: transformed_points = processing.coordinates[body_part]
        # affine transform to match model arena
        transformed_points = np.matmul(np.append(processing.registration_data[0], np.zeros((1, 3)), 0),
                                       np.concatenate((transformed_points[0:1], transformed_points[1:2],
                                                       np.ones((1, len(transformed_points[0])))), 0))
        # fill in the coordinates array with the transformed points
        processing.coordinates[body_part][0] = transformed_points[0, :]
        processing.coordinates[body_part][1] = transformed_points[1, :]
        # fill in the coordinates array with the uncertain points as nan
        processing.coordinates[body_part][0][nan_index] = np.nan
        processing.coordinates[body_part][1][nan_index] = np.nan
        # plot distance from origin, if applicable
        if plot: ax.plot(processing.coordinates[body_part][0][:10000]**2 +  processing.coordinates[body_part][1][:10000]**2)
    if plot:
        ax.set_title('Distance from origin, 1st 10000 timepoints')
        ax.legend(body_parts)
        plt.pause(2)



def compute_speed_position_angles(processing):
    # get frame size
    width, height = processing.registration_data[-1][0], processing.registration_data[-1][0]
    # get the body parts used
    opened_dlc_config_file = open(processing.dlc_config_file)
    parsed_dlc_config_file = yaml.load(opened_dlc_config_file)
    body_parts = parsed_dlc_config_file['bodyparts']
    # array of all body parts, axis x body part x frame
    all_body_parts = np.zeros((2, len(body_parts), processing.coordinates[body_parts[0]].shape[1]))
    for i, body_part in enumerate(body_parts):
        all_body_parts[:, i, :] = processing.coordinates[body_part][0:2]
    # make sure values are within the proper range
    all_body_parts[all_body_parts >= width] = width - 1
    # compute particular body part locations by taking the nan median of several points
    body_part_locations(all_body_parts, processing.coordinates)
    # compute speed
    delta_position = np.concatenate( ( np.zeros((2,1)), np.diff(processing.coordinates['center_location']) ) , axis = 1)
    processing.coordinates['speed'] = np.sqrt(delta_position[0,:]**2 + delta_position[1,:]**2)
    # linearly interpolate any remaining nan values
    locations = ['speed', 'snout_location', 'head_location', 'neck_location', 'center_body_location', 'center_location', 'butt_location']
    for loc_num, loc in enumerate(locations):
        if 'speed' in loc:
            processing.coordinates[loc] = np.array(pd.Series(processing.coordinates[loc]).interpolate())
            processing.coordinates[loc] = np.array(pd.Series(processing.coordinates[loc]).fillna(method='bfill'))
            processing.coordinates[loc] = np.array(pd.Series(processing.coordinates[loc]).fillna(method='ffill'))
        else:
            for i in [0,1]:
                processing.coordinates[loc][i] = np.array(pd.Series(processing.coordinates[loc][i]).interpolate())
                processing.coordinates[loc][i] = np.array(pd.Series(processing.coordinates[loc][i]).fillna(method='bfill'))
                processing.coordinates[loc][i] = np.array(pd.Series(processing.coordinates[loc][i]).fillna(method='ffill'))
    # compute angles
    processing.coordinates['body_angle'] = np.angle((processing.coordinates['neck_location'][0] - processing.coordinates['butt_location'][0]) + (-processing.coordinates['neck_location'][1] + processing.coordinates['butt_location'][1]) * 1j, deg=True)
    processing.coordinates['head_angle'] = np.angle((processing.coordinates['snout_location'][0] - processing.coordinates['neck_location'][0]) + (-processing.coordinates['snout_location'][1] + processing.coordinates['neck_location'][1]) * 1j, deg=True)

    # correct locations out of frame
    locations = ['head_location', 'snout_location', 'neck_location', 'center_body_location', 'center_location', 'butt_location']
    for loc in locations:
        processing.coordinates[loc][0][processing.coordinates[loc][0] >= width ] = width - 1
        processing.coordinates[loc][1][processing.coordinates[loc][1] >= height] = height - 1
        processing.coordinates[loc][0][processing.coordinates[loc][0] < 0] = 0
        processing.coordinates[loc][1][processing.coordinates[loc][1] < 0] = 0




