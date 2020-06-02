import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from eval import load_net
from map_handler import load_map, plot_map


def pad_to_preserve_shape(inputs, network_width):
    """
    Pad inputs to account for shrinkage at the map borders due to the input
    shape of the network.
    """

    # wrap to recover output size
    pad_before = int(network_width / 2)
    pad_after = int(network_width / 2) - 1 + (network_width % 2)

    return np.pad(inputs, ((pad_before, pad_after), (pad_before, pad_after)), 'edge')


def rotate_map(map_image, angle):
    """
    Rotate map_image by angle (in degrees).
    """

    # image dimensions and center
    (h, w) = map_image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # calculate rotation matrix
    rot_mat = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)

    # perform the actual rotation and return the image
    return cv2.warpAffine(map_image, rot_mat, (w, h), flags=cv2.INTER_NEAREST,
                          borderValue=0.5)


def get_pad_for_rotate(image):
    """
    Pad map to make it square and add enough border to avoid cutting relevant
    parts when rotating it.
    """

    max_dim = np.max(image.shape)
    new_l = np.sqrt(2) * max_dim

    pad_side_x = int((new_l - image.shape[0]) // 2)
    pad_side_y = int((new_l - image.shape[1]) // 2)

    return (pad_side_x, pad_side_x), (pad_side_y, pad_side_y)


def get_pad_for_downsample(inputs, factor):
    """
    Pad input to make sure it is dividable by factor.
    """

    width, height = inputs.shape
    width_pad = 0
    height_pad = 0

    if width % factor != 0:
        width_pad = factor - width % factor
    if height % factor != 0:
        height_pad = factor - height % factor

    padding_shape = np.array([[int(width_pad / 2), int(width_pad / 2 + width_pad % 2)],
                              [int(height_pad / 2), int(height_pad / 2 + height_pad % 2)]])
    return padding_shape


def prepare_map(input_map, downsample_factor):
    """
    Pad map for augmentation: account for rotation and downsampling.
    """

    # 1. pad for rotation
    rot_padding = get_pad_for_rotate(input_map)
    rot_padded = np.pad(input_map, rot_padding, 'edge')

    # 2. pad to for dividable by downsample_factor
    four_padding = get_pad_for_downsample(rot_padded, downsample_factor)
    my_map_pad = np.pad(rot_padded, four_padding, 'edge')

    # pad image
    padding = rot_padding + four_padding

    return my_map_pad, padding


def segment_map(map_image, net, device, num_angles):
    """
    Segment map_image by processing it with net at num_angles equidistant angles.
    """

    map_in, padding = prepare_map(map_image, net.downsample_factor)
    augmented_map = np.zeros(map_in.shape)

    with torch.no_grad():

        for i in range(num_angles):
            # rotate map
            rot = 360.0 / num_angles * i
            map_in_rot = rotate_map(map_in, rot)

            # pad map to account for border shrinkage due to network input size
            map_rot_pad = pad_to_preserve_shape(map_in_rot, net.net_size)

            # process rotated map
            map_rot_pad = map_rot_pad.reshape([1, 1, map_rot_pad.shape[0], map_rot_pad.shape[1]])
            image = torch.from_numpy(map_rot_pad).float().to(device)
            pred_rot = torch.nn.functional.softmax(net(image), dim=1)

            # upsample output (if necessary)
            pred_rot_resized = torch.nn.functional.interpolate(
                pred_rot, scale_factor=net.downsample_factor, mode='bilinear', align_corners=False)
            pred_rot_resized = pred_rot_resized.cpu().numpy()[0, 1, :, :]

            # rotate back
            pred = rotate_map(pred_rot_resized, -rot)

            # remember maximum predicted occupancy
            augmented_map = np.maximum(augmented_map, pred)

    # transfer unknown space from input
    for i in range(augmented_map.shape[0]):
        for j in range(augmented_map.shape[1]):
            if map_in[i, j] == 0.5:
                augmented_map[i, j] = map_in[i, j]

    # cut augmented map to original shape
    augmented_map = augmented_map[padding[0, 0]:-padding[0, 1],
                                  padding[1, 0]:-padding[1, 1]]

    return augmented_map


def save_segmented_map(network_dir, segmented_map_image, map_name):
    map_save_dir = os.path.join(network_dir, "segmented")

    if not os.path.exists(map_save_dir):
        os.mkdir(map_save_dir)

    map_save_path = os.path.join(map_save_dir, map_name)

    seg_map_image = (1.0 - segmented_map_image) * 254.0
    seg_map_image = np.flip(seg_map_image.T, axis=0)

    cv2.imwrite(map_save_path, seg_map_image)
    print("saved segmented image to: ", map_save_path)


if __name__ == '__main__':
    # Call segment_map.py from command line, with the map_yaml and network_dir as
    # mandatory arguments.

    parser = argparse.ArgumentParser(
        description='Segment an occupancy map.')
    parser.add_argument('map_yaml',
                        help='input map yaml file')
    parser.add_argument('network_dir',
                        help='network directory')
    parser.add_argument('--num_angles', type=int, default=8,
                        help='number of equidistant angles, default: 8')
    parser.add_argument('--save', action='store_true',
                        help='save segmented map image')

    args = parser.parse_args()

    # load map
    map_im, _, _, map_occ_name = load_map(args.map_yaml, False)

    # load net
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = load_net(args.network_dir, torch_device)

    segmented_map = segment_map(map_im, network, torch_device, args.num_angles)

    if args.save:
        save_segmented_map(args.network_dir, segmented_map, map_occ_name)

    plot_map(segmented_map)
    plt.waitforbuttonpress()
