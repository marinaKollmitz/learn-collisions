import argparse
import torch
from eval import load_net, get_classification_scores
import os
import yaml
import numpy as np
from map_handler import load_map
from segment_map import segment_map, save_segmented_map

package_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


def evaluate_maps(segmented_maps, gt_collision_maps):
    """
    Calculate classification scores from segmented maps and GT collision maps.
    """

    all_predictions = []
    all_labels = []

    for i in range(len(segmented_maps)):
        segmented_map = segmented_maps[i]
        collision_map = gt_collision_maps[i]

        # flatten maps for pixel-wise labels and predictions
        labels = collision_map.flatten()
        predictions = segmented_map.flatten()

        # remove unknown areas from predictions and labels
        valid_indices = (labels != 0.5) & (predictions != 0.5)

        labels = labels[valid_indices]
        predictions = predictions[valid_indices]

        all_labels.extend(labels)
        all_predictions.extend(predictions)

    return get_classification_scores(np.array(all_predictions), np.array(all_labels))


def evaluate_segmentation(network_dir, num_angles, dataset_name, split, save_maps=False):
    """
    Evaluate segmentation performance of network.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load network with trained parameters
    net = load_net(network_dir, device)

    # collect occupancy maps
    gt_collision_maps = []
    segmented_maps = []

    dataset_path = os.path.join(package_dir, "datasets/" + str(dataset_name))
    splits_file_path = os.path.join(dataset_path, "data_split.yml")
    splits = yaml.load(open(splits_file_path, 'r'), Loader=yaml.SafeLoader)

    set_envs = splits[split]

    for env in set_envs:
        print("processing %s ..." % env)

        occupancy_map_yaml = os.path.join(dataset_path, "occupancy_maps/" + str(env) + "_occ.yaml")
        collision_map_yaml = os.path.join(dataset_path, "collision_maps/" + str(env) + "_col.yaml")

        # segment occupancy map
        occ_map_image = load_map(occupancy_map_yaml, False)[0]
        segmented_map = segment_map(occ_map_image, net, device, num_angles)
        segmented_maps.append(segmented_map)

        # load GT collision map
        collision_map_image = load_map(collision_map_yaml, False)[0]
        gt_collision_maps.append(collision_map_image)

        if save_maps:
            save_segmented_map(network_dir, segmented_map, env + "_occ.pgm")

    return evaluate_maps(segmented_maps, gt_collision_maps)


if __name__ == '__main__':
    # call evaluate_segmentation.py from command line, with the network_dir
    # as mandatory argument
    parser = argparse.ArgumentParser(
        description='Evaluate collision pose classification performance of a network.')
    parser.add_argument('network_dir',
                        help='network directory')
    parser.add_argument('--dataset', default='SceneNetCollision',
                        help='name of dataset to be evaluated, default: SceneNetCollision')
    parser.add_argument('--split', default='test',
                        help='split to be evaluated, default: test')
    parser.add_argument('--num_angles', type=int, default=8,
                        help='number of equidistant angles, default: 8')
    parser.add_argument('--save', action='store_true',
                        help='save segmented map image')

    args = parser.parse_args()
    evaluate_segmentation(args.network_dir, args.num_angles, args.dataset, args.split,
                          save_maps=args.save)
