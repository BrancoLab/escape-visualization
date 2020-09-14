# escape-visualization
 DLC tracking, video registration, escape video rendering
 
 ### Python environment
 An environment file is included, which can be cloned to match the libraries used when running this code. It could take at least an hour to download all libraries.
 
 ### setup_parameters.py
 Parameters and analysis steps to run are found in this file. If libraries are installed correctly, this should run immediately on the sample data. 
 **options.do_DLC_tracking** DeepLabCut tracking. options.dlc_config_file points to the network location. I can provide this trained network if requested. If running the sample data set, just set options.do_DLC_tracking to False to avoid complicated installation issues.
 **options.do_registration** Registers behavior video the a common reference point across videos. Includes fisheye lens correction if applicable.
 **options.do_coordinate_processing** Aligns the DeepLabCut coordinates to the common reference frame and processes them (e.g. median filter)
 **options.do_visualization** Visualizes images and movies of the escape and renderings thereof. This typically runs slightly slower than real time.
 
 ### arena_drawings.py
 Either use OpenCV drawing functions or import an outside file of the arena - This is used for registration and for visualization.
 
 ### locate_body_parts.py
 If you do not use the same 13 body parts during DeepLapCut tracking, that is ok -- modify this file so that the same locations on the animal are nonetheless being used for analysis.
