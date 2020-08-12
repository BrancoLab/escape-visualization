from setup_parameters import setup
import pandas as pd
import os
import cv2
import pickle
from termcolor import colored
from auxiliary_code.registration import registration
from auxiliary_code.dlc_funcs import extract_dlc_coordinates, median_filter_and_transform_coordinates, compute_speed_position_angles
from auxiliary_code.escape_visualization import visualize_escape

class visualize_escapes():
    def __init__(self):
        setup(self) # import settings from setup_lite.py
        if self.do_DLC_tracking:
            print(colored(' - Performing DLC Tracking', 'green')); self.run_DLC_tracking()
        if self.do_registration:
            print(colored(' - Registering videos or loading registration data', 'green')); self.run_registration()
        if self.do_coordinate_processing:
            print(colored(' - Extracting or loading coordinates', 'green')); self.run_coordinate_processing()
        if self.do_visualization:
            print(colored(' - Visualizing escapes', 'green')); self.run_visualization()

    def run_DLC_tracking(self):
        from deeplabcut.pose_estimation_tensorflow import analyze_videos
        for video_path in self.videos: analyze_videos(self.dlc_config_file, video_path)

    def run_registration(self):
        for v, video_path in enumerate(self.videos):
            registration_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '_registration_data')
            # check if registration data already exists
            if os.path.isfile(registration_file_name) and not self.overwrite_saved_registration:
                print(os.path.basename(registration_file_name) + ' already exists')
            else: # generate and save a new registration if one does not exist
                print('registering ' + video_path)
                registration_data = registration(self, video_path, v)
                with open(registration_file_name, "wb") as dill_file: pickle.dump(registration_data, dill_file)

    def run_coordinate_processing(self):
        for v, video_path in enumerate(self.videos):
            coordinates_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '_coordinates')
            registration_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '_registration_data')
            # check if extracted coordinates already exists
            if os.path.isfile(coordinates_file_name) and not self.overwrite_saved_processing:
                print(os.path.basename(coordinates_file_name) + ' already exists')
            else:  # generate and save a new extracted coordinates file if needed
                print('extracting coordinates for ' + video_path)
                with open(registration_file_name, "rb") as dill_file: self.registration_data = pickle.load(dill_file)
                self.coordinates = extract_dlc_coordinates(self.dlc_config_file, video_path)
                median_filter_and_transform_coordinates(self, v)
                compute_speed_position_angles(self)
                with open(coordinates_file_name, "wb") as dill_file: pickle.dump(self.coordinates, dill_file)

    def run_visualization(self):
        for v, video_path in enumerate(self.videos):
            print('visualizing escapes for ' + video_path)
            coordinates_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '_coordinates')
            registration_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + '_registration_data')
            # load coordinates and registration
            with open(coordinates_file_name, "rb") as dill_file: self.coordinates = pickle.load(dill_file)
            with open(registration_file_name, "rb") as dill_file: self.registration_data = pickle.load(dill_file)
            # visualize escape
            visualize_escape(self, video_path, v)

if __name__ == "__main__":
    d = visualize_escapes()
