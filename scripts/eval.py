import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os
import yaml
import torch

from networks.conv_lenet import FCNLeNet
from networks.dilated_conv_lenet import DilatedFCNLeNet


def load_net(network_dir, device):

    # find network input size from config
    config_path = os.path.join(network_dir, "config.yml")
    config = yaml.load(open(config_path, 'r'), yaml.SafeLoader)
    input_size = config['input_size']
    model_type = config['model_type']

    # initialize network
    if model_type == 'fcn_lenet':
        net = FCNLeNet(input_size)

    elif model_type == 'dilated_fcn_lenet':
        net = DilatedFCNLeNet(input_size)

    else:
        print("model %s not supported" % model_type)
        return None

    # load network params
    model_path = os.path.join(network_dir, "model.pkl")
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)

    return net


def get_classification_scores(predictions, labels, plot=True):
    collision_thresh = 0.5
    binary_predictions = predictions > collision_thresh

    accuracy = metrics.accuracy_score(labels, binary_predictions)
    precision = metrics.precision_score(labels, binary_predictions)
    recall = metrics.recall_score(labels, binary_predictions)
    average_precision = metrics.average_precision_score(labels, predictions)

    print('accuracy score: %f' % accuracy)
    print('precision: %f, recall: %f' % (precision, recall))
    print('average precision score: %f' % average_precision)

    if plot:
        # plot precision recall curve
        plt.figure()
        precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))

        plt.waitforbuttonpress()

    return precision, recall, average_precision
