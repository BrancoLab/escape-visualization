import cv2
import numpy as np
import scipy.misc
import imageio
import os
import warnings
from auxiliary_code.registration import model_arena
warnings.filterwarnings("ignore")



def visualize_escape(self, video_path, v):
    # open the raw behavior video
    raw_video = open_video_and_get_properties(self, video_path)
    # set up model arena
    arena, _, shelter_ROI = model_arena((self.height, self.width), self.arena_type[v])
    if len(arena.shape)<3: arena = cv2.cvtColor(arena, cv2.COLOR_GRAY2RGB)  # give it 3 color channels rather than 1
    # set up multi-trial video
    multi_trial_background_image = arena.copy()
    multi_trial_visualization_clip = set_up_multi_trial_video(self, video_path)
    # loop over each trial
    for trial, start_stim_end_frame in enumerate(self.start_stim_end_frames[v]):
        # get frame numbers and fisheye correction and initialize videos
        self.start_frame, self.stim_frame, self.end_frame = start_stim_end_frame[0], start_stim_end_frame[1], start_stim_end_frame[2]
        map1, map2 = load_fisheye_correction(self)
        video_clip, visualization_clip = set_up_video_clips(self, trial, video_path)
        # initialize arenas for mouse mask
        arrived_at_shelter, smoothed_speed, trial_plot, model_mouse_mask_initial, last_darkened_mouse_silhouette = \
            initialize_visualization_vars(arena, self)
        # set the video to the start frame
        raw_video.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        '''     Loop over each frame, making the videos and images       '''
        while True:
            # get the frame
            ret, frame = raw_video.read()
            #get the frame number
            frame_num = int(raw_video.get(cv2.CAP_PROP_POS_FRAMES)) - 1 # count starting from zero
            frames_past_stimulus = frame_num - self.stim_frame
            # apply the registration and fisheye correction
            if map1.size: frame = register_frame(frame, self, map1, map2, v)
            else: frame = frame[:,:,0]
            # prior to stimulus onset, refresh frame to initialized frame
            if frames_past_stimulus < 0:
                visualization_video_frame = arena.copy()
                multi_trial_visualization_frame = multi_trial_background_image.copy()
            # extract DLC coordinates and make a model mouse mask
            mouse_silhouette, large_ellipse_around_mouse, body_location = get_mouse_silhouette(self.coordinates, frame_num, model_mouse_mask_initial,
                                                                                               scale = self.mouse_silhouette_size)
            # use speed to determine model mouse coloration
            speed_color_light, speed_color_dark = speed_colors(smoothed_speed[frame_num], self.speed_thresholds)
            # at the stimulus onset, get the mouse's contour and store its position
            if frames_past_stimulus == 0:
                _, contours, _ = cv2.findContours(mouse_silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x_position_at_stim, y_position_at_stim = float(body_location[0]), float(body_location[1])
            # if the mouse is far from the last dark mouse silhouette (or pre-stimulus), make another one
            if frames_past_stimulus <= 0 or (np.sum(large_ellipse_around_mouse * last_darkened_mouse_silhouette) == 0 and not arrived_at_shelter):
                # add dark mouse to the video arenas
                visualization_video_frame[mouse_silhouette.astype(bool)] = visualization_video_frame[mouse_silhouette.astype(bool)] * speed_color_dark
                multi_trial_visualization_frame[mouse_silhouette.astype(bool)] = multi_trial_visualization_frame[mouse_silhouette.astype(bool)] * speed_color_dark
                last_darkened_mouse_silhouette = mouse_silhouette
            # continuous shading after stimulus onset
            elif frames_past_stimulus > 0:
                # detect whether the mouse has is close to the shelter
                if (np.sum(shelter_ROI * mouse_silhouette) and self.stop_at_shelter) or arrived_at_shelter:
                    # end video in 2 seconds after arriving at shelter
                    if not arrived_at_shelter: arrived_at_shelter = True; self.end_frame = frame_num + self.fps * 2
                else: # add light mouse silhouette to video arenas
                    visualization_video_frame[mouse_silhouette.astype(bool)] = visualization_video_frame[mouse_silhouette.astype(bool)] * speed_color_light
                    multi_trial_visualization_frame[mouse_silhouette.astype(bool)] = multi_trial_visualization_frame[mouse_silhouette.astype(bool)] * speed_color_light
            # add red trail to the previous trials' arena
            if frames_past_stimulus > 0 and not arrived_at_shelter:
                dist_from_start = np.sqrt((x_position_at_stim - float(body_location[0]))**2 + (y_position_at_stim - float(body_location[1]))**2)
                previous_trial_color = np.array([.98, .98, .98]) + np.max((0,self.dist_to_make_red - dist_from_start))/self.dist_to_make_red * np.array([-.02, -.02, .02])
                multi_trial_background_image[mouse_silhouette.astype(bool)] = multi_trial_background_image[mouse_silhouette.astype(bool)] * previous_trial_color
            # redraw the white contour on each frame after the stimulus
            if frame_num >= self.stim_frame:
                cv2.drawContours(visualization_video_frame, contours, 0, (255, 255, 255), thickness = 1, lineType = 8)
                cv2.drawContours(multi_trial_visualization_frame, contours, 0, (255, 255, 255), thickness=1, lineType=8)
            # add a looming spot - for actual loom
            if self.show_loom_in_video_clip: frame = show_loom_in_video_clip(frame, frame_num, self, trial_plot, visualization_video_frame)
            # display current frames
            cv2.imshow(video_path + ' clip', frame);
            cv2.imshow(video_path + ' visualization', visualization_video_frame)
            cv2.imshow(video_path + ' multi-trial visualization', multi_trial_visualization_frame)
            # show as fast as possible; press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            # write current frame to videos
            video_clip.write(frame)
            visualization_clip.write(visualization_video_frame)
            multi_trial_visualization_clip.write(multi_trial_visualization_frame)
            # end video
            if frame_num >= self.end_frame: break
        video_clip.release(); visualization_clip.release()
        # save trial images
        escape_image_file_name = os.path.join(os.path.dirname(video_path), 'escape image ' + str(trial+1) + '.tif')
        imageio.imwrite(escape_image_file_name, visualization_video_frame[:,:,::-1])
        if trial:
            multi_trial_escape_image_file_name = os.path.join(os.path.dirname(video_path), 'escape image ' + str(trial+1) + ' + previous trials.tif')
            imageio.imwrite(multi_trial_escape_image_file_name, multi_trial_visualization_frame[:,:,::-1])
        # draw red silhouette for previous trials arena
        cv2.drawContours(multi_trial_background_image, contours, 0, (100, 100, 255), thickness=-1)
    raw_video.release(); multi_trial_visualization_clip.release()


def show_loom_in_video_clip(frame, frame_num, self, trial_plot, video_arena):
    # get the radius of the expanding circle
    stim_radius = 30 * (frame_num - self.stim_frame) * ((frame_num - self.stim_frame) < 10) * (frame_num > self.stim_frame)
    if stim_radius:
        # copy frame for superposition with loom spot
        frame = frame.copy() # needed for some reason
        loom_frame = frame.copy()
        # draw loom on image
        stimulus_location = tuple(self.coordinates['center_body_location'][:, self.stim_frame - 1].astype(np.uint16))
        cv2.circle(loom_frame, stimulus_location, stim_radius, 100, -1)
        # show loom spot
        alpha = .3
        cv2.addWeighted(frame, alpha, loom_frame, 1 - alpha, 0, frame)
    return frame


def initialize_visualization_vars(arena, self):
    # initialize more quantities
    trial_plot = arena.copy()
    frames_past_stimulus = 0
    arrived_at_shelter = False
    smoothed_speed = np.concatenate((np.zeros(6 - 1), np.convolve(self.coordinates['speed'], np.ones(12), mode='valid'), np.zeros(6))) / 12
    model_mouse_mask_initial = np.zeros(arena.shape[0:2]).astype(np.uint8)
    last_darkened_mouse_silhouette = 0
    return arrived_at_shelter, smoothed_speed, trial_plot, model_mouse_mask_initial, last_darkened_mouse_silhouette


def load_fisheye_correction(self):
    # load fisheye mapping if applicable
    if self.registration_data[3]:
        maps = np.load(self.fisheye_correction_file);
        map1 = maps[:, :, 0:2];
        map2 = maps[:, :, 2] * 0
    else:
        map1 = np.array([]); map2 = np.array([])
    return map1, map2

def set_up_multi_trial_video(self, video_path):
    # set up the escape clips for saving
    multi_trial_visualization_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + ' multi-trial visualization.mp4v')
    fourcc = cv2.VideoWriter_fourcc(*"MPEG") # video compression codec
    multi_trial_visualization_clip = cv2.VideoWriter(multi_trial_visualization_file_name, fourcc, self.fps, (self.width, self.height), True)
    return multi_trial_visualization_clip

def set_up_video_clips(self, trial, video_path):
    # set up the escape clips for saving
    visualization_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + ' visualization ' + str(trial+1) + '.mp4v')
    video_file_name = os.path.join(os.path.dirname(video_path), os.path.basename(video_path)[:-4] + ' clip ' + str(trial+1) + '.mp4v')
    fourcc = cv2.VideoWriter_fourcc(*"MPEG") # video compression codec
    video_clip = cv2.VideoWriter(video_file_name, fourcc, self.fps, (self.width, self.height), False)
    visualization_clip = cv2.VideoWriter(visualization_file_name, fourcc, self.fps, (self.width, self.height), True)
    return video_clip, visualization_clip

def open_video_and_get_properties(self, video_path):
    # open the behaviour video and get its properties
    raw_video = cv2.VideoCapture(video_path)
    self.fps = raw_video.get(cv2.CAP_PROP_FPS)
    self.height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    self.width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    return raw_video


def get_mouse_silhouette(coordinates, frame_num, model_mouse_mask_initial, scale = 16):
    '''     extract DLC coordinates and make a model mouse mask     '''
    # extract coordinates
    body_angle = coordinates['body_angle'][frame_num]
    head_angle = coordinates['head_angle'][frame_num]
    body_head_angle = (body_angle * 2 + head_angle * 1) / 3
    head_body_angle = (body_angle * 1 + head_angle * 2) / 3
    body_location = tuple(coordinates['center_body_location'][:, frame_num].astype(np.uint16))
    head_location = tuple(coordinates['neck_location'][:, frame_num].astype(np.uint16))
    body_head_location = tuple(((np.array(body_location) * 2 + np.array(head_location) * 1) / 3).astype(int))
    head_body_location = tuple(((np.array(body_location) * 1 + np.array(head_location) * 2) / 3).astype(int))


    # draw ellipses representing model mouse
    mouse_silhouette = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(scale * .9), int(scale * .5)), 180 - body_angle, 0, 360, 100, thickness=-1)
    mouse_silhouette = cv2.ellipse(mouse_silhouette, body_head_location, (int(scale * .5), int(scale * .33)), 180 - body_head_angle, 0, 360, 100, thickness=-1)
    mouse_silhouette = cv2.ellipse(mouse_silhouette, head_body_location, (int(scale * .7), int(scale * .35)), 180 - head_body_angle, 0, 360, 100, thickness=-1)
    mouse_silhouette = cv2.ellipse(mouse_silhouette, head_location, (int(scale * .6), int(scale * .3)), 180 - head_angle, 0, 360, 100, thickness=-1)

    # make a single large ellipse used to determine when do use the flight_color_dark
    large_ellipse_around_mouse = cv2.ellipse(model_mouse_mask_initial.copy(), body_location, (int(scale * 2.5), int(scale * 1.5)), 180 - body_angle, 0, 360, 100, thickness=-1)
    return mouse_silhouette, large_ellipse_around_mouse, body_location



def register_frame(frame, self, map1, map2, v):
    '''     go from a raw to a registered frame        '''
    x_offset, y_offset = self.offset[v][0], self.offset[v][1]
    # make into 2D
    frame_register = frame[:, :, 0]
    # fisheye correction
    if map1.size:
        # pad the frame
        frame_register = cv2.copyMakeBorder(frame_register, y_offset, int((map1.shape[0] - frame.shape[0]) - y_offset),
                                            x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)
        # fisheye correct the frame
        frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        # un-pad the frame
        frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                         x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]
    # register the frame
    frame = cv2.cvtColor(cv2.warpAffine(frame_register, self.registration_data[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)
    # make into 2D again
    frame = frame[:, :, 0]
    return frame


def speed_colors(speed, speed_thresholds):
    '''    set up colors for speed-dependent DLC analysis    '''
    # colors depending on speed
    slow_color = np.array([254, 254, 254])
    medium_color = np.array([254, 253.5, 252.64])
    fast_color = np.array([254, 250, 240])
    super_fast_color = np.array([254,  250, 200])
    # apply thresholds
    if speed > speed_thresholds[2]:
        speed_color = super_fast_color
    elif speed > speed_thresholds[1]:
        speed_color = ((speed_thresholds[2] - speed) * fast_color + (speed - speed_thresholds[1]) * super_fast_color) / (speed_thresholds[2] - speed_thresholds[1])
    elif speed > speed_thresholds[0]:
        speed_color = ((speed_thresholds[1] - speed) * medium_color + (speed - speed_thresholds[0]) * fast_color) / (speed_thresholds[1] - speed_thresholds[0])
    else:
        speed_color = (speed * medium_color + (speed_thresholds[0] - speed) * slow_color) / speed_thresholds[0]
    # turn this color into a color multiplier
    speed_color_light = 1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .08)
    speed_color_dark = (1 - (1 - speed_color / [255, 255, 255]) / (np.mean(1 - speed_color / [255, 255, 255]) / .38) ) **2

    return speed_color_light, speed_color_dark