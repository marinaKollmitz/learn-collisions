import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from collision_dataset import CollisionDataset
from eval import get_classification_scores, load_net


def get_predictions(model, dataset, torch_device):
    """
    Get classification predictions of network model on dataset.
    """
    data_loader = DataLoader(dataset, batch_size=128, num_workers=0)

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            if i % 10 == 0:
                print("evaluating batch: %3d of %3d" % (i, int(len(dataset) / data_loader.batch_size)))

            inputs = data['image']
            labels = data['label']
            labels = labels.numpy()

            inputs = inputs.to(torch_device)
            predictions = torch.nn.functional.softmax(model(inputs), dim=1).cpu().numpy()

            all_labels.extend(labels)
            all_predictions.extend(predictions[:, 1, 0, 0])

    return np.array(all_predictions), np.array(all_labels)


def evaluate_collisions(network_dir, dataset_name, split):
    """
    Evaluate collision classification scores for network.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load network with trained parameters
    net = load_net(network_dir, device)

    print("loading dataset %s, split %s ..." % (dataset_name, split))
    val_set = CollisionDataset(dataset_name, split, net.net_size)

    predictions, labels = get_predictions(net, val_set, device)
    get_classification_scores(predictions, labels)


if __name__ == '__main__':
    # call evaluate_collisions.py from command line, with the network_dir
    # as mandatory argument
    parser = argparse.ArgumentParser(
        description='Evaluate collision pose classification performance of a network.')
    parser.add_argument('network_dir',
                        help='network directory')
    parser.add_argument('--dataset', default='SceneNetCollision',
                        help='name of dataset to be evaluated, default: SceneNetCollision')
    parser.add_argument('--split', default='validation',
                        help='split to be evaluated, default: validation')

    args = parser.parse_args()
    evaluate_collisions(args.network_dir, args.dataset, args.split)
