import numpy as np
import cv2
from termcolor import colored

def model_arena(shape, arena_type = 'circle with shelter'):
    '''     GENERATE A MODEL ARENA IMAGE        '''

    # generate arena reference image for each arena type
    if arena_type == 'circle with shelter':
        '''     this arena is 92 cm, so this reference image is scaled at 1 cm / 10 pixels        '''
        # initialize model arena
        model_arena = 255 * np.ones((1000, 1000)).astype(np.uint8)
        # arena outline
        cv2.circle(model_arena, (500, 500), 460, 0, 1, lineType=16)
        # shelter
        cv2.rectangle(model_arena, (500 - 50, 500 + 408 - 50), (500 + 50, 500 + 408 + 50), 150, thickness=-1)
        # shelter ROI (this can be used during visualization to stop the video once the mouse reaches the shelter)
        shelter_ROI = model_arena * 0
        cv2.rectangle(shelter_ROI, (500 - 50, 500 + 408 - 50), (500 + 50, 500 + 408 + 50), 255, thickness=-1)

        # add points for the user to click during registration (at least 4)
        click_points = np.array(([500, 500 + 460], [500 - 460, 500], [500, 500 - 460], [500 + 460, 500]))  # four corners

    elif arena_type == 'backward 7 maze':
        '''     arbitrary dimensions picked from convience; not sure of scaling to real distances        '''
        # initialize model arena
        model_arena = cv2.imread('C:\\Users\\SWC\\Desktop\\the aMAZEing fede\\fede schematic.png')
        # shelter ROI (this can be used during visualization to stop the video once the mouse reaches the shelter)
        shelter_ROI = model_arena[:,:,0] * 0 # should be 2D not 3D
        cv2.rectangle(shelter_ROI, (470-50, 200-30), (470+50, 200+30), 255, thickness=-1)

        # add points for the user to click during registration (at least 4)
        click_points = np.array(([470, 220], [85, 200], [290, 420], [480, 635]))

    # resize model arena to size of video
    if model_arena.shape[1] / model_arena.shape[0] != shape[1] / shape[0]:
        print(colored('Model and video aspect ratio do not agree -- resizing model arena', 'red'))
    click_points = click_points * [shape[1] / model_arena.shape[1], shape[0] / model_arena.shape[0]]
    model_arena = cv2.resize(model_arena, shape[::-1])
    shelter_ROI = cv2.resize(shelter_ROI, shape[::-1], interpolation=cv2.INTER_NEAREST)

    # show arena
    # cv2.imshow('arena',model_arena)

    return model_arena, click_points, shelter_ROI