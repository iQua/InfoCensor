import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

import sys
sys.path.insert(1, '../')
from models.encoder import *
from models.decoder import *
from models.discriminator import *

from data_preprocess.load_health import load_health
from data_preprocess.load_utkface import load_utkface
from data_preprocess.load_nlps import load_twitter
from data_preprocess.load_german import load_german

from data_preprocess.config import dataset_class_params

Dloss = nn.BCELoss()

def load_attack_data(args):
    if args.dataset == 'german':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        train_size=args.attacker_size,
                                                                                                        batch_size=args.batch_size)
        assert input_dim == dataset_class_params['german']['input_dim']


    if args.dataset == 'health':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_health(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        train_size=args.attacker_size,
                                                                                                        binarize=False,
                                                                                                        batch_size=args.batch_size)
        assert input_dim == dataset_class_params['health']['input_dim']


    elif args.dataset == 'utkface':
        train_loader, test_loader, img_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr=args.target_attr,
                                                                                                        sensitive_attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        train_size=args.attacker_size,
                                                                                                        batch_size=args.batch_size)
        assert img_dim == dataset_class_params['utkface']['img_dim']

    elif args.dataset == 'twitter':
        train_loader, test_loader, voc_size, target_num_classes, sensitive_num_classes = load_twitter(sensitive_attr=args.sensitive_attr,
                                                                                                      random_seed=args.seed,
                                                                                                      train_size=args.attacker_size,
                                                                                                      batch_size=args.batch_size)
        assert voc_size == dataset_class_params['twitter']['voc_size']

    return train_loader, test_loader

def load_feature_learner(args, info_model=False):
    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    elif args.dataset == 'utkface':
        img_dim = dataset_class_params['utkface']['img_dim']
        feature_learner = lenet_encoder(img_dim=img_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    elif args.dataset == 'twitter':
        voc_size = dataset_class_params['twitter']['voc_size']
        feature_learner = lstm_encoder(voc_size=voc_size, embedding_dim=32,
                                       out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    return feature_learner

def load_decoder(args, info_model=False):
    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        decoder = mlp_decoder(in_dim=args.num_features,
                              out_dim=input_dim,
                              drop_rate=args.drop_rate)


    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        decoder = mlp_decoder(in_dim=args.num_features,
                              out_dim=input_dim,
                              drop_rate=args.drop_rate)


    elif args.dataset == 'utkface':
        img_dim = dataset_class_params['utkface']['img_dim']
        decoder = lenet_decoder(input_dim=args.num_features,
                                img_dim=img_dim,
                                drop_rate=args.drop_rate)

    elif args.dataset == 'twitter':
        voc_size = dataset_class_params['twitter']['voc_size']
        decoder = lstm_decoder(voc_size=voc_size, fea_dim=args.num_features,
                            hidden_dim=args.num_features, drop_rate=args.drop_rate)

    return decoder



def load_discriminator(args):

    if args.dataset == 'german':
        input_dim = dataset_class_params['german']['input_dim']
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model) ## learn q(z|x)

    if args.dataset == 'health':
        input_dim = dataset_class_params['health']['input_dim']
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model) ## learn q(z|x)

    elif args.dataset == 'utkface':
        img_dim = dataset_class_params['utkface']['img_dim']
        feature_learner = lenet_encoder(img_dim=img_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    elif args.dataset == 'twitter':
        voc_size = dataset_class_params['twitter']['voc_size']
        feature_learner = lstm_encoder(voc_size=voc_size, embedding_dim=32,
                                       out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

    return feature_learner


def data_generator(data_loader):
    while True:
        for _, (x, y, s) in enumerate(data_loader):
            yield (x, y, s)


def discriminator_loss(discriminator, real_output, fake_output):

    real = torch.ones((real_output.size(0), ), device=real_output.device).float()

    fake = torch.zeros((fake_output.size(0), ), device=real_output.device).float()

    loss = Dloss(discriminator(real_output), real) + Dloss(discriminator(fake_output), fake)

    return loss

def generator_loss(discriminator, fake_output):

    real = torch.ones((fake_output.size(0), ), device=fake_output.device).float()

    loss = Dloss(discriminator(fake_output), real)

    return loss
