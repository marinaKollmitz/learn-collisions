import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from collision_dataset import CollisionDataset
from networks.conv_lenet import FCNLeNet
from networks.dilated_conv_lenet import DilatedFCNLeNet
from evaluate_collisions import get_predictions

plt.ioff()


def adjust_learning_rate(optimizer, lr):
    """
    Adjusts the learning rate.
    """

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_collision_classifier(model_type, patch_size, dataset_name):
    """
    Train network collision classifier on dataset.
    """

    # training config
    config = {'learning_rate': 0.025,
              'learning_rate_step': 0.5,
              'batch_size': 32,
              'num_epochs': 10,
              'class_weights': [1, 2],
              'input_size': patch_size,
              'model_type': model_type}

    # path to save the network
    package_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
    date_str = datetime.datetime.now().strftime("-%Y-%m-%d_%H:%M:%S")
    network_path = os.path.join(package_dir, "networks/" + str(model_type) + "-" + str(patch_size) + date_str)
    os.mkdir(network_path)

    # save config to file
    yaml.dump(config, open(os.path.join(network_path, "config.yml"), 'w'),
              yaml.SafeDumper)

    # initialize network
    if model_type == 'fcn_lenet':
        net = FCNLeNet(patch_size)

    elif model_type == 'dilated_fcn_lenet':
        net = DilatedFCNLeNet(patch_size)

    else:
        print("model not supported")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)

    # load data
    print("loading datasets ...")
    train_set = CollisionDataset(dataset_name, "train", patch_size)
    val_set = CollisionDataset(dataset_name, "validation", patch_size)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'],
                              shuffle=True, num_workers=8)

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.from_numpy(np.array(config['class_weights'])).float().to(device))
    optimizer = torch.optim.SGD(net.parameters(), lr=config["learning_rate"])

    losses = []
    fig_loss = plt.figure()
    ax_loss = fig_loss.add_subplot(1, 1, 1)

    aps = []
    fig_aps = plt.figure()
    ax_aps = fig_aps.add_subplot(1, 1, 1)

    for epoch in range(config['num_epochs']):

        running_loss = 0.0
        new_lr = config['learning_rate'] * (config['learning_rate_step']) ** epoch
        print("setting learning rate to: ", new_lr)
        adjust_learning_rate(optimizer, new_lr)

        for i, data in enumerate(train_loader, 0):
            inputs = data['image'].to(device)
            labels = data['label'].to(device)

            optimizer.zero_grad()
            net.zero_grad()

            outputs = net(inputs)
            # output shape is [batches, classes, 1, 1]

            # make sure that we have the output shape matches: 2 classes
            # and 1 output neuron for a patch input
            assert outputs.shape[1] == 2 and outputs.shape[3] == 1

            outputs = outputs[:, :, 0, 0]

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                losses.append(running_loss / 1000)

                ax_loss.clear()
                ax_loss.plot(losses)
                ax_loss.set_title("loss vs iteration")
                ax_loss.set_xlabel("iteration")
                ax_loss.set_ylabel("loss")

                fig_loss.savefig(os.path.join(network_path, 'loss.png'))

                running_loss = 0.0

        # save model parameters
        model_path = os.path.join(network_path, "model.pkl")
        print("saving network to", model_path)
        torch.save(net.state_dict(), model_path)

        print("evaluating validation performance")

        # evaluate performance on validation set after each epoch
        with torch.no_grad():

            predictions, labels = get_predictions(net, val_set, device)
            average_precision = average_precision_score(labels, predictions)

            print("average precision score: ", average_precision)
            aps.append(average_precision)

            ax_aps.clear()
            ax_aps.plot(aps)
            ax_aps.set_title("average precision vs epoch")
            ax_aps.set_xlabel("epoch")
            ax_aps.set_ylabel("average precision")

            fig_aps.savefig(os.path.join(network_path, 'aps.png'))

    print('Finished Training')


if __name__ == '__main__':
    """
    Call learn_collisions.py from command line, with the patch_size as 
    mandatory argument.
    """

    parser = argparse.ArgumentParser(
        description='Train a collision classification network.')
    parser.add_argument('patch_size', type=int,
                        help='size of map input patch')
    parser.add_argument('--model', default='dilated_fcn_lenet',
                        help='model architecture, default: dilated_fcn_lenet')
    parser.add_argument('--dataset', default='SceneNetCollision',
                        help='name of dataset to train on, default: SceneNetCollision')

    args = parser.parse_args()
    train_collision_classifier(args.model, args.patch_size, args.dataset)
