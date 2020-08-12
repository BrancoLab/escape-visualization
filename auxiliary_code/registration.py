import cv2
import numpy as np
import scipy.misc
from tqdm import tqdm
import glob
import os
import pandas as pd
from termcolor import colored
from arena_drawings import model_arena

def registration(self, video_path, v):
    ''' CONTROL BACKGROUND ACQUISITION AND ARENA REGISTRATION    '''
    # initialize data structure
    session = pd.Series();
    # Get background by averaging over frames in the video
    background, width, height = get_background(video_path,start_frame=300, avg_over=20)
    # Register arena
    registration_data = register_arena(background, width, height, self.fisheye_correction_file, self.offset[v], self.arena_type[v])
    registration_data.append([width, height])

    return registration_data





def get_background(vidpath, start_frame = 1000, avg_over = 100):
    '''
    ..........................EXTRACT BACKGROUND BY AVERAGING FRAMES THROGHOUT THE VIDEO................................
    '''
    # initialize the video
    vid = cv2.VideoCapture(vidpath)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background = np.zeros((height, width))
    num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    vid.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # initialize the counters
    every_other = int(num_frames / avg_over)
    j = 0

    # loop through num_frames frames
    for i in tqdm(range(num_frames)):
        # only use every other x frames
        if i % every_other == 0:
            vid.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = vid.read()
            # store the current frame in as a numpy array
            if ret:
                background += frame[:, :, 0]
                j+=1

    # normalize the background intensity to the number of frames summed
    background = (background / (j)).astype(np.uint8)

    # show the background
    cv2.imshow('background', background)
    cv2.waitKey(10)

    # release the video
    vid.release()

    return background, width, height




def register_arena(background, width, height, fisheye_correction, offset, arena_type):
    '''
    ..........................GUI TO REGISTER ARENAS TO COMMON FRAMEWORK................................
    '''

    # create model arena and background
    arena, click_points, _ = model_arena((height, width), arena_type = arena_type)

    # get x and y offeset
    x_offset, y_offset = offset[0], offset[1]

    # load the fisheye correction
    if os.path.isfile(str(fisheye_correction)):
        maps = np.load(fisheye_correction)
        map1 = maps[:, :, 0:2]
        map2 = maps[:, :, 2]*0

        background_copy = cv2.copyMakeBorder(background, y_offset, int((map1.shape[0] - background.shape[0]) - y_offset),
                                             x_offset, int((map1.shape[1] - background.shape[1]) - x_offset), cv2.BORDER_CONSTANT, value=0)

        background_copy = cv2.remap(background_copy, map1, map2, interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        background_copy = background_copy[y_offset:-int((map1.shape[0] - background.shape[0]) - y_offset),
                          x_offset:-int((map1.shape[1] - background.shape[1]) - x_offset)]
    else:
        background_copy = background.copy()
        fisheye_correction = ''
        print('fisheye correction not found')

    # initialize clicked points
    if len(arena.shape)==3: blank_arena = cv2.cvtColor(arena, cv2.COLOR_RGB2GRAY)
    else: blank_arena = arena.copy()
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]

    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)

    make_new_transform_immediately = True

    print('\nSelect reference points on the experimental background image in the indicated order')

    # initialize clicked point arrays
    background_data = [background_copy, np.array(([], [])).T]
    arena_data = [[], np.array(([], [])).T]

    # add 1-2-3-4 markers to model arena
    for i, point in enumerate(click_points.astype(np.uint32)):
        arena = cv2.circle(arena, (point[0], point[1]), 3, 255, -1)
        arena = cv2.circle(arena, (point[0], point[1]), 4, 0, 1)
        cv2.putText(arena, str(i+1), tuple(point), 0, .55, 50, thickness=2)
        cv2.putText(arena, str(i + 1), tuple(point), 0, .55, 255, thickness=1)

        point = np.reshape(point, (1, 2))
        arena_data[1] = np.concatenate((arena_data[1], point))

    # initialize GUI
    cv2.startWindowThread()
    cv2.namedWindow('background')
    cv2.imshow('background', background_copy)
    cv2.namedWindow('model arena')
    cv2.imshow('model arena', arena)

    # create functions to react to clicked points
    cv2.setMouseCallback('background', select_transform_points, background_data)  # Mouse callback

    while True: # take in clicked points until four points are clicked
        cv2.imshow('background',background_copy)

        number_clicked_points = background_data[1].shape[0]
        if number_clicked_points == len(arena_data[1]):
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # perform projective transform
    M = cv2.estimateRigidTransform(background_data[1], arena_data[1], False)


    # REGISTER BACKGROUND, BE IT WITH LOADED OR CREATED TRANSFORM
    registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])

    # --------------------------------------------------
    # overlay images
    # --------------------------------------------------
    alpha = .7
    colors = [[150, 0, 150], [0, 255, 0]]
    color_array = make_color_array(colors, background.shape)

    registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                             * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
    arena_color = (cv2.cvtColor(blank_arena, cv2.COLOR_GRAY2RGB)
                   * np.squeeze(color_array[:, :, :, 1])).astype(np.uint8)

    overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
    cv2.namedWindow('registered background')
    cv2.imshow('registered background', overlaid_arenas)

    # --------------------------------------------------
    # initialize GUI for correcting transform
    # --------------------------------------------------
    print('\nIteratively left click model arena & Right click model background // Press ''y'' when finished')
    print('Green means the model arena is brighter, purple means the photo is brighter')

    update_transform_data = [overlaid_arenas,background_data[1], arena_data[1], M, background_copy]

    # create functions to react to additional clicked points
    cv2.setMouseCallback('registered background', additional_transform_points, update_transform_data)

    # take in clicked points until 'q' is pressed
    initial_number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
    M_initial = M

    while True:
        cv2.imshow('registered background',overlaid_arenas)
        cv2.imshow('background', registered_background)
        number_clicked_points = [update_transform_data[1].shape[0], update_transform_data[2].shape[0]]
        update_transform = False
        k = cv2.waitKey(10)
        # If a left and right point are clicked:
        if number_clicked_points[0]>initial_number_clicked_points[0] and number_clicked_points[1]>initial_number_clicked_points[1]:
            initial_number_clicked_points = number_clicked_points
            # update transform and overlay images
            try:
                M = cv2.estimateRigidTransform(update_transform_data[1], update_transform_data[2],False) #True ~ full transform
                update_transform = True
            except:
                continue
            update_transform = True
        elif  k == ord('r'):
            print('Transformation erased')
            update_transform_data[1] = np.array(([],[])).T
            update_transform_data[2] = np.array(([],[])).T
            initial_number_clicked_points = [3,3]
        elif k == ord('q') or k == ord('y'):
            print('Registration completed')
            break

        if update_transform:
            update_transform_data[3] = M

            registered_background = cv2.warpAffine(background_copy, M, background.shape[::-1])
            registered_background_color = (cv2.cvtColor(registered_background, cv2.COLOR_GRAY2RGB)
                                           * np.squeeze(color_array[:, :, :, 0])).astype(np.uint8)
            overlaid_arenas = cv2.addWeighted(registered_background_color, alpha, arena_color, 1 - alpha, 0)
            update_transform_data[0] = overlaid_arenas

    cv2.destroyAllWindows()
    return [M, update_transform_data[1], update_transform_data[2], fisheye_correction]


# mouse callback function I
def select_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 3, 255, -1)
        data[0] = cv2.circle(data[0], (x, y), 4, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[1] = np.concatenate((data[1], clicks))

# mouse callback function II
def additional_transform_points(event,x,y, flags, data):
    if event == cv2.EVENT_RBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (200,0,0), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        M_inverse = cv2.invertAffineTransform(data[3])
        transformed_clicks = np.matmul(np.append(M_inverse,np.zeros((1,3)),0), [x, y, 1])
        # M_inverse = np.linalg.inv(data[3])
        # M_inverse = cv2.findHomography(data[2][:len(data[1])], data[1])
        # transformed_clicks = np.matmul(M_inverse[0], [x, y, 1])

        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 2, (0, 0, 200), -1)
        data[4] = cv2.circle(data[4], (int(transformed_clicks[0]), int(transformed_clicks[1])), 3, 0, 1)

        clicks = np.reshape(transformed_clicks[0:2],(1,2))
        data[1] = np.concatenate((data[1], clicks))
    elif event == cv2.EVENT_LBUTTONDOWN:

        data[0] = cv2.circle(data[0], (x, y), 2, (0,200,200), -1)
        data[0] = cv2.circle(data[0], (x, y), 3, 0, 1)

        clicks = np.reshape(np.array([x, y]),(1,2))
        data[2] = np.concatenate((data[2], clicks))

def make_color_array(colors, image_size):
    color_array = np.zeros((image_size[0],image_size[1], 3, len(colors)))  # create coloring arrays
    for c in range(len(colors)):
        for i in range(3):  # B, G, R
            color_array[:, :, i, c] = np.ones((image_size[0],image_size[1])) * colors[c][i] / sum(
                colors[c])
    return color_array


def register_frame(frame, x_offset, y_offset, registration, map1, map2):
    '''
    ..........................GO FROM A RAW TO A REGISTERED FRAME................................
    '''
    frame_register = frame[:, :, 0]

    frame_register = cv2.copyMakeBorder(frame_register, y_offset,
                                        int((map1.shape[0] - frame.shape[0]) - y_offset),
                                        x_offset, int((map1.shape[1] - frame.shape[1]) - x_offset),
                                        cv2.BORDER_CONSTANT, value=0)
    frame_register = cv2.remap(frame_register, map1, map2, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    frame_register = frame_register[y_offset:-int((map1.shape[0] - frame.shape[0]) - y_offset),
                     x_offset:-int((map1.shape[1] - frame.shape[1]) - x_offset)]



    frame = cv2.cvtColor(cv2.warpAffine(frame_register, registration[0], frame.shape[0:2]),cv2.COLOR_GRAY2RGB)

    return frame



def invert_fisheye_map(registration, inverse_fisheye_correction):
    '''
    ..........................GO FROM A NORMAL FISHEYE MAP TO AN INVERTED ONE.........................
    '''

    if len(registration) == 5:
        pass
    elif os.path.isfile(inverse_fisheye_correction):
        registration.append(inverse_fisheye_correction)
    elif len(registration) == 4:  # setup fisheye correction
        print('creating inverse fisheye map')
        inverse_maps = np.load(registration[3])
        # invert maps
        inverse_maps[inverse_maps < 0] = 0

        maps_x_orig = inverse_maps[:, :, 0]
        maps_x_orig[maps_x_orig > 1279] = 1279
        maps_y_orig = inverse_maps[:, :, 1]
        maps_y_orig[maps_y_orig > 1023] = 1023

        map_x = np.ones(inverse_maps.shape[0:2]) * np.nan
        map_y = np.ones(inverse_maps.shape[0:2]) * np.nan
        for x in range(inverse_maps.shape[1]):
            for y in range(inverse_maps.shape[0]):
                map_x[maps_y_orig[y, x], maps_x_orig[y, x]] = x
                map_y[maps_y_orig[y, x], maps_x_orig[y, x]] = y

        grid_x, grid_y = np.mgrid[0:inverse_maps.shape[0], 0:inverse_maps.shape[1]]
        valid_values_x = np.ma.masked_invalid(map_x)
        valid_values_y = np.ma.masked_invalid(map_y)

        valid_idx_x_map_x = grid_x[~valid_values_x.mask]
        valid_idx_y_map_x = grid_y[~valid_values_x.mask]

        valid_idx_x_map_y = grid_x[~valid_values_y.mask]
        valid_idx_y_map_y = grid_y[~valid_values_y.mask]

        map_x_interp = interpolate.griddata((valid_idx_x_map_x, valid_idx_y_map_x), map_x[~valid_values_x.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)
        map_y_interp = interpolate.griddata((valid_idx_x_map_y, valid_idx_y_map_y), map_y[~valid_values_y.mask],
                                            (grid_x, grid_y), method='linear').astype(np.uint16)

        fisheye_maps_interp = np.zeros((map_x_interp.shape[0], map_x_interp.shape[1], 2)).astype(np.uint16)
        fisheye_maps_interp[:, :, 0] = map_x_interp
        fisheye_maps_interp[:, :, 1] = map_y_interp

        np.save('C:\\Drive\\DLC\\transforms\\inverse_fisheye_maps.npy', fisheye_maps_interp)

    return registration