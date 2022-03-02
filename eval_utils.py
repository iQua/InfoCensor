import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
import sys
import logging

## models
sys.path.insert(1, '../')
from models.classifier import *
from models.encoder import *
from models.decoder import *

## dataset
from data_preprocess.load_health import load_health
from data_preprocess.load_utkface import load_utkface
from data_preprocess.load_nlps import load_twitter
from data_preprocess.load_german import load_german

## utils
from utils import (fairness_metric,
                   spd_metric)

logger = logging.getLogger(__name__)



def evaluate(args, info_model=True):

    if args.dataset == 'german':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, info_model=info_model) ## learn q(z|x)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)


    if args.dataset == 'health':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_health(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        binarize=False,
                                                                                                        batch_size=args.batch_size)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, info_model=info_model) ## learn q(z|x)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    elif args.dataset == 'utkface':
        train_loader, test_loader, img_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr=args.target_attr,
                                                                                                        sensitive_attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)

        feature_learner = lenet_encoder(img_dim=img_dim, out_dim=args.num_features, info_model=info_model)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    elif args.dataset == 'twitter':
        train_loader, test_loader, voc_size, target_num_classes, sensitive_num_classes = load_twitter(sensitive_attr=args.sensitive_attr,
                                                                                                      random_seed=args.seed,
                                                                                                      batch_size=args.batch_size)

        feature_learner = lstm_encoder(voc_size=voc_size, embedding_dim=32,
                                       out_dim=args.num_features, info_model=info_model)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    ### use cuda
    if args.use_cuda:
        feature_learner, normal_classifier = feature_learner.cuda(), normal_classifier.cuda()

    ### set to train mode
    feature_learner.eval()
    normal_classifier.eval()

    state_dict = torch.load(os.path.join(args.out_dir, args.model_name + '_feature_model.pth'))
    feature_learner.load_state_dict(state_dict)
    state_dict = torch.load(os.path.join(args.out_dir, args.model_name + '_classifier_model.pth'))
    normal_classifier.load_state_dict(state_dict)

    criterion = nn.CrossEntropyLoss()

    test_normal_loss = 0
    test_normal_acc = 0
    test_n = 0

    ## record the fairness metric
    sensitive_class_count = {}
    for slabel in range(sensitive_num_classes):
        sensitive_class_count[slabel] = [0 for tlabel in range(target_num_classes)]

    for i, (X, y, s) in enumerate(test_loader):
        if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
        z = feature_learner(X)
        pred = normal_classifier(z)
        celoss = criterion(pred, y)

        pred_labels = pred.max(1)[1]
        for idx in range(s.size(0)):
            sensitive_class_count[s[idx].item()][pred_labels[idx].item()] += 1

        test_n += y.size(0)
        test_normal_acc += (pred.max(1)[1] == y).sum().item()
        test_normal_loss += celoss.item()*y.size(0)
    # print(sensitive_class_count)
    sensitive_class_freq, fairMI = fairness_metric(sensitive_class_count, sensitive_num_classes)
    max_spd, avg_spd = spd_metric(sensitive_class_freq, sensitive_num_classes)

    logger.info('Normal Test Acc: %.4f \t Normal Test Loss: %.4f', test_normal_acc/test_n, test_normal_loss/test_n)
    print('Normal Test Acc: {0:.4f} \t Normal Test Loss: {0:.4f}'.format(test_normal_acc/test_n, test_normal_loss/test_n))

    for slabel in range(sensitive_num_classes):
        class_log_info = '['
        for i in range(target_num_classes-1):
            class_log_info += str(sensitive_class_freq[slabel][i]) + ', '
        class_log_info += str(sensitive_class_freq[slabel][target_num_classes-1]) + ']'
        logger.info('sensitive class: %d:\t %s', slabel, class_log_info)

    logger.info('fairness metric: %.6f', fairMI)
    logger.info('Max SPD: %.6f, Avg SPD: %.6f', max_spd, avg_spd)
    print('fairness metric: {0:.6f}, Max SPD: {0:.6f}, Avg SPD: {0:.6f}'.format(fairMI, max_spd, avg_spd))


def store_representations(args, info_model=True):
    if args.dataset == 'german':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, info_model=info_model) ## learn q(z|x)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)


    if args.dataset == 'health':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_health(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        binarize=False,
                                                                                                        batch_size=args.batch_size)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, info_model=info_model) ## learn q(z|x)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    elif args.dataset == 'utkface':
        train_loader, test_loader, img_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr=args.target_attr,
                                                                                                        sensitive_attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)

        feature_learner = lenet_encoder(img_dim=img_dim, out_dim=args.num_features, info_model=info_model)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    elif args.dataset == 'twitter':
        train_loader, test_loader, voc_size, target_num_classes, sensitive_num_classes = load_twitter(sensitive_attr=args.sensitive_attr,
                                                                                                      random_seed=args.seed,
                                                                                                      batch_size=args.batch_size)

        feature_learner = lstm_encoder(voc_size=voc_size, embedding_dim=32,
                                       out_dim=args.num_features, info_model=info_model)
        normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes)

    ### use cuda
    if args.use_cuda:
        feature_learner, normal_classifier = feature_learner.cuda(), normal_classifier.cuda()

    ### set to train mode
    feature_learner.eval()
    normal_classifier.eval()

    state_dict = torch.load(os.path.join(args.out_dir, args.model_name + '_feature_model.pth'))
    feature_learner.load_state_dict(state_dict)
    state_dict = torch.load(os.path.join(args.out_dir, args.model_name + '_classifier_model.pth'))
    normal_classifier.load_state_dict(state_dict)

    representations, ylabels, slabels = [], [], []

    for i, (X, y, s) in enumerate(test_loader):
        if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
        z = feature_learner(X)
        representations.append(z.detach().cpu().numpy())
        ylabels.append(y.detach().cpu().numpy())
        slabels.append(s.detach().cpu().numpy())


    representations = np.concatenate(representations, axis=0)
    ylabels = np.concatenate(ylabels, axis=0)
    slabels = np.concatenate(slabels, axis=0)

    np.savez(os.path.join(args.out_dir, args.model_name + '_representations.npz'), representations, ylabels, slabels, allow_pickle=True)
