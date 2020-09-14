import os
'''     setup the parameters for tracking, registration, coordinate extraction, and visualization      '''
def setup(options):
    '''     SELECT DATA (User suggested to programatically streamline these parameters instead of inputting manually)   '''
    # select videos -- currently using the sample video included in the repo
    options.videos = [os.path.join(os.path.dirname(__file__), 'sample data', 'escapes without obstacle.avi')]
    # provide timing parameters for each video -- [[trial 1 stim frame,trial 2 stim frame],[**session 2 frame #s]]
    options.stim_frames = [[16325, 24801, 30455]]
    options.pre_stim_duration = 3 # in seconds
    options.max_escape_duration = 12 # in seconds
    # set the arena of the common coordinate behavior reference frame - you need to input this arena in model_arena() in registration_lite.py
    options.arena_type = ['circle with shelter' for x in range(len(options.videos))]

    '''    SELECT WHAT WE ARE DOING (need to do the earlier steps at some point before the later steps)   '''
    options.do_DLC_tracking = False # For demo purposes, keep this False! The output is included in the repo
    options.do_registration = True
    options.do_coordinate_processing = True
    options.do_visualization = True

    options.overwrite_saved_registration = False # if you need to redo the registration step
    options.overwrite_saved_processing = False # if you need to redo the processing step

    '''     DLC OPTIONS    '''
    options.dlc_config_file = 'D:\\data\\DLC_nets\\Barnes-Philip-2018-11-22\\config.yaml' # generated in the DLC pipeline
    options.show_extracted_coordinates_plot = True
    options.median_filter_duration_in_frames = 7 # (7 in my well labeled setup)
    options.minimum_confidence_from_dlc = 0 # if under this, set tracking coordinate to nan and interpolate (.999 in my well labeled setup)
    options.maximum_error_drom_dlc = 60 # accept tracking in point is below this distance (in pixels) from the rest of the points

    '''     FISHEYE CORRECTION PARAMETERS       '''
    # 'fisheye correction\\fisheye_maps.npy' and 'fisheye correction\\inverse_fisheye_maps.npy' are included in repo for our standard camera
    # However, if you do not want to apply a fisheye correction, write None for the file names
    options.fisheye_correction_file = os.path.join(os.path.dirname(__file__), 'fisheye correction', 'fisheye_maps.npy') #None
    options.inverted_fisheye_correction_file = os.path.join(os.path.dirname(__file__), 'fisheye correction', 'inverse_fisheye_maps.npy') #None
    options.offset = [[300, 120] for x in range(len(options.videos))] # for fisheye correction; [0,0] if the frames are cropped from the full camera image size

    '''     VISUALIZATION PARAMETERS        '''
    options.stop_at_shelter = True
    options.show_loom_in_video_clip = True
    options.show_body_part_dots = False
    options.mouse_silhouette_size = 16 # how big is the mouse in the rendering
    options.speed_thresholds = [7.2, 10.8, 14.4] # for coloration by speed, in pixel / frame; corresponds to 30 / 45 / 60 cm/s in my setup
    options.dist_to_make_red = 150 # leave a red trace for previous escapes


