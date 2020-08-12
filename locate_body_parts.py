import numpy as np

def body_part_locations(all_body_part_locations, coordinates):
    '''     From the clicked DLC points, calculate the locations of body parts of interest

    Given the ordered list of body parts you've selected for DLC tracking, take the mean of
    those points in order to compute the location of the snout, head, neck, butt, center of the body,
    and center of the mouse
    '''

    coordinates['snout_location'] = np.nanmean(all_body_part_locations[:, 0:3, :], axis=1)
    coordinates['head_location'] = np.nanmean(all_body_part_locations[:, 0:6, :], axis=1)
    coordinates['neck_location'] = np.nanmean(all_body_part_locations[:, 3:9, :], axis=1)
    coordinates['butt_location'] = np.nanmean(all_body_part_locations[:, 6:, :], axis=1)
    coordinates['center_body_location'] = np.nanmean(all_body_part_locations[:, 6:12, :], axis=1)
    coordinates['center_location'] = np.nanmean(all_body_part_locations[:, :, :], axis=1)