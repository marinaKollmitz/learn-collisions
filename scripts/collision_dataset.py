import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from map_handler import MapHandler
from posefile_parser import parse_pose_file

random.seed(123)
package_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))


class CollisionDataset(Dataset):

    def __init__(self, dataset_name, split, image_size):

        self.collision_examples = []

        cache_dir = os.path.join(package_dir, "data_cache")
        if not os.path.isdir(cache_dir):
            os.mkdir(cache_dir)

        # for saving pickled images
        cache_identifier = str(dataset_name) + str(split) + str(image_size)
        cache_path = os.path.join(cache_dir, cache_identifier + ".pkl")

        if os.path.exists(cache_path):
            # load cache
            print("cached database exists, loading: ", cache_path)
            self.collision_examples = torch.load(cache_path)

        else:
            # load examples and save to pickle file
            print("no cached database found, generating collision examples. This will take a while ...")
            dataset_path = os.path.join(package_dir, "datasets/" + str(dataset_name))
            splits_file_path = os.path.join(dataset_path, "data_split.yml")
            splits = yaml.load(open(splits_file_path, 'r'), Loader=yaml.SafeLoader)

            set_envs = splits[split]
            poses_dir = os.path.join(dataset_path, "collision_poses")
            maps_dir = os.path.join(dataset_path, "occupancy_maps")

            self.load_examples(set_envs, poses_dir, maps_dir, image_size)

            print("saving databased to ", cache_path)
            torch.save(self.collision_examples, cache_path)

    def load_examples(self, set_envs, poses_dir, maps_dir, image_size):

        for i, env in enumerate(set_envs):
            print("processing: %3d from %3d, %s" % (i, len(set_envs), env))

            # pose and map file of environment
            pose_file_path = os.path.join(poses_dir, str(env) + ".txt")
            map_path = os.path.join(maps_dir, str(env) + "_occ.yaml")

            # map_handler for cutting the map patches
            map_handler = MapHandler(map_path, image_size, thresholded_map=False)

            # parse the pose file
            pose_list = parse_pose_file(pose_file_path, map_path, map_handler)
            random.shuffle(pose_list)

            # generate map patches with labels
            for collision_pose in pose_list:

                pose = collision_pose['pose']
                collision = collision_pose['collision']

                image = map_handler.cut_patch(pose[0], pose[1], pose[2])

                if image is not None:

                    if collision:
                        label = 1
                    else:
                        label = 0

                    sample = {'image': torch.from_numpy(np.array([image])).float(),
                              'label': torch.tensor(label).long(),
                              'pose': pose}

                    self.collision_examples.append(sample)

        print("total number of collision examples", len(self.collision_examples))

    def __len__(self):
        return len(self.collision_examples)

    def __getitem__(self, idx):
        sample = self.collision_examples[idx]
        return sample
