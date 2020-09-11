import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable


def read_map_image(map_image_path, thresholded_map):
    """
    Read map image to numpy array with cell probabilities between 0 and 1.
    Use thresholded_map=True if the map image file uses thresholded values for
    the occupancies, like ROS occupancy maps. If the occupancy map contains
    non-thresholded occupancy values between 0-254, set thresholded_map to
    False.
    """

    # load image picture
    map_image = cv2.imread(map_image_path, -1)

    # flip and transpose array to organize axis: x=bottom and y=left
    map_image = np.flip(map_image, axis=0)
    map_image = map_image.T

    occ_map = np.zeros(map_image.shape)

    # translate from pixel values to occupancy probabilities
    if thresholded_map:

        # verify that image is thresholded
        occ_map[(map_image != MapHandler.PGM_OCCUPIED) &
                (map_image != MapHandler.PGM_FREE) &
                (map_image != MapHandler.PGM_UNKNOWN)] = np.nan

        if np.isnan(occ_map).any():
            print("occupancy value not valid, is the map file a ROS occupancy map?")
            return None

        occ_map[map_image == MapHandler.PGM_OCCUPIED] = MapHandler.PROB_OCCUPIED
        occ_map[map_image == MapHandler.PGM_FREE] = MapHandler.PROB_FREE
        occ_map[map_image == MapHandler.PGM_UNKNOWN] = MapHandler.PROB_UNKNOWN

    else:
        occ_threshold = 0.25  # 0.25 threshold in gmapping
        occ_map = (254.0 - map_image) / 254.0
        occ_map[occ_map < occ_threshold] = MapHandler.PROB_FREE
        occ_map[occ_map == 0.5] = MapHandler.PROB_UNKNOWN
        occ_map[(occ_map >= 0.25) & (occ_map != 0.5)] = MapHandler.PROB_OCCUPIED

    return occ_map


def load_map(map_yaml_path, thresholded_map):
    """
    Load map image specified in map_yaml_path file and return it together with
    the map_resolution and map_origin.
    """

    try:
        map_info = yaml.load(open(map_yaml_path, 'r'), Loader=yaml.SafeLoader)
        map_resolution = map_info['resolution']
        map_origin = map_info['origin']

        map_dir = os.path.dirname(map_yaml_path)
        map_name = map_info['image']
        map_image_path = os.path.join(map_dir, map_name)
        occ_map = read_map_image(map_image_path, thresholded_map)

        return occ_map, map_resolution, map_origin, map_name

    except yaml.YAMLError as err:
        print("MapHandler, could not parse yaml file: ", err)


def plot_map(map_array, color_map_name="jet"):
    """
    Plot map cell probabilities colored according to color_map_name. The unknown
    space is plotted in grey for clarity.
    """

    map_array = map_array.T

    # get cell probability colors, make unknown space transparent so it can
    # shine through in Grey
    color_map = plt.get_cmap(color_map_name)
    alphas = np.zeros(map_array.shape)
    alphas[map_array != 0.5] = 1
    prob_colors = color_map(map_array)
    prob_colors[..., -1] = alphas

    ax = plt.gca()

    # plot map_array in color_map for colorbar
    im = ax.imshow(map_array, color_map, origin='lower',
                   extent=(0, map_array.shape[1], 0, map_array.shape[0]))

    # plot the map in black and white, so the unknown space can shine through
    ax.imshow(map_array, "Greys", origin='lower', interpolation='none',
              extent=(0, map_array.shape[1], 0, map_array.shape[0]))

    # plot the cell probabilities
    ax.imshow(prob_colors, origin='lower', interpolation='none',
              extent=(0, map_array.shape[1], 0, map_array.shape[0]))

    # plot the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    plt.colorbar(im, cax=cax)


class MapHandler:
    # occupancy values in pgm map used by ros
    PGM_OCCUPIED = 0
    PGM_FREE = 254
    PGM_UNKNOWN = 205

    # occupancy probabilities
    PROB_OCCUPIED = 1.0
    PROB_FREE = 0.0
    PROB_UNKNOWN = 0.5

    def __init__(self, map_yaml_path, patch_size, thresholded_map=True):
        """
        MapHandler for working with 2D maps. Main purpose is cutting out map
        patches, according to the patch_size. Use thresholded_map=True if the
        map image file, specified in map_yaml_path, uses thresholded values for
        the occupancies, like ROS occupancy maps. If the occupancy map contains
        non-thresholded occupancy values between 0-254, set thresholded_map to
        False.
        """

        # read map file into array
        self.map, self.map_resolution, self.map_origin = load_map(map_yaml_path,
                                                                  thresholded_map)[0:3]
        self.patch_size = patch_size

    def get_map(self):
        return self.map

    def world_to_cell(self, world_x, world_y):
        """
        Transform from (continuous) world coordinates to (continuous) cell
        coordinates.
        """

        cell_x = (world_x - self.map_origin[0]) / self.map_resolution
        cell_y = (world_y - self.map_origin[1]) / self.map_resolution

        return cell_x, cell_y

    def inside_map(self, cell_x, cell_y):
        """
        Check if cell coordinates are within map bounds.
        """

        if 0 < cell_x < self.map.shape[0] and 0 < cell_y < self.map.shape[1]:
            return True
        else:
            return False

    def cut_patch(self, x, y, theta, plot=False):
        """
        Cut out map patch centered around the pose S=(x,y,theta). Use plot=True
        for visualization and debugging.
        """

        # center pixel for which we want to make predictions later
        center_ind = int(self.patch_size / 2)
        window_indices = np.linspace(0, self.patch_size - 1, num=self.patch_size) - center_ind

        patch = np.zeros([self.patch_size, self.patch_size])
        map_x, map_y = self.world_to_cell(x, y)

        # fill patch by looking up the transformed coordinates in the map
        for i, window_x in enumerate(window_indices):
            for j, window_y in enumerate(window_indices):
                # rotated window coordinates
                rotated_x = np.cos(theta) * window_x - np.sin(theta) * window_y
                rotated_y = np.sin(theta) * window_x + np.cos(theta) * window_y

                x = map_x + rotated_x
                y = map_y + rotated_y

                map_val = MapHandler.PROB_UNKNOWN
                if self.inside_map(x, y):
                    map_val = self.map[int(x), int(y)]

                patch[i, j] = map_val

        # visualization for debugging
        if plot:
            # plot the map with the cutout window
            fig, (ax1, ax2) = plt.subplots(1, 2)
            plt.sca(ax1)
            plot_map(self.map, color_map_name="Greys")
            dx = 4.0 * np.cos(theta)
            dy = 4.0 * np.sin(theta)

            ax1.plot(map_x, map_y, "r*")
            ax1.arrow(map_x, map_y, dx, dy, width=0.3, color='r')

            # plot bbox as polygon
            polygon = []
            x_min = window_indices[0]
            x_max = window_indices[-1]
            y_min = window_indices[0]
            y_max = window_indices[-1]

            polygon.append([map_x + (np.cos(theta) * x_min - np.sin(theta) * y_min),
                            map_y + (np.sin(theta) * x_min + np.cos(theta) * y_min)])
            polygon.append([map_x + (np.cos(theta) * x_max - np.sin(theta) * y_min),
                            map_y + (np.sin(theta) * x_max + np.cos(theta) * y_min)])
            polygon.append([map_x + (np.cos(theta) * x_max - np.sin(theta) * y_max),
                            map_y + (np.sin(theta) * x_max + np.cos(theta) * y_max)])
            polygon.append([map_x + (np.cos(theta) * x_min - np.sin(theta) * y_max),
                            map_y + (np.sin(theta) * x_min + np.cos(theta) * y_max)])

            pol = Polygon(polygon, fill=False, edgecolor='r')
            ax1.add_artist(pol)

            # plot the patch itself
            plt.sca(ax2)
            plot_map(patch, color_map_name="Greys")

        return patch


if __name__ == '__main__':
    package_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    map_yaml = os.path.join(package_dir,
                            "datasets/SceneNetCollision/occupancy_maps/1Bathroom_1_labels_occ.yaml")

    map_cutout = MapHandler(map_yaml, 20, thresholded_map=False)

    map_cutout.cut_patch(2.245, 0.0, -0.365100, plot=True)
    plt.waitforbuttonpress()
