import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
import time
import os
import sys
import logging
import random
from attack_utils import load_attack_data, load_feature_learner
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
## models
sys.path.insert(1, '../')
from models.classifier import *
from models.encoder import *

from data_preprocess.config import dataset_class_params
## dataset
# from data_preprocess.load_health import load_health
# from data_preprocess.load_utkface import load_utkface
# from data_preprocess.load_nlps import load_twitter
# from data_preprocess.load_german import load_german

import argparse
if torch.cuda.is_available():
    import setGPU

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--attacker-size', default=0.5, type=float)

    parser.add_argument('--dataset', default='health', type=str, choices=['german', 'health', 'utkface', 'twitter'])
    parser.add_argument('--num-epochs', default=100, type=int)

    parser.add_argument('--num-features', default=128, type=int)

    parser.add_argument('--target-attr', default='age', type=str, choices=['age', 'gender', 'credit', 'charlson'])
    parser.add_argument('--sensitive-attr', default='none', type=str, choices=['none', 'age', 'gender', 'race', 'identity'])
    parser.add_argument('--surrogate', default='ce', type=str)

    ## follow the existing work to train gan, we use adam optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument('--beta1', type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--beta2', type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--drop-rate', type=float, default=0.1, help='dropout rate to train the auxiliary model')
    parser.add_argument('--adv-drop-rate', type=float, default=0.0, help='dropout rate to train the adversary model')


    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--out-dir', default='decensor', type=str)
    parser.add_argument('--defense-method', default='adv_censor', type=str)
    parser.add_argument('--defense-model-name', default=None, type=str)
    parser.add_argument('--model-name', default=None, type=str)

    parser.add_argument('--seed', default=1, type=int) ## the seed is very likely to differ from the seed used by the defender

    return parser.parse_args()


def main(args):
    info_bool = ('info' in args.defense_method) or ('vae' in args.defense_method)
    if 'no_rand' in args.defense_method: info_bool = False
    print('Is info model or not? {}'.format(info_bool))

    train_loader, test_loader = load_attack_data(args)

    sensitive_num_classes = dataset_class_params[args.dataset][args.sensitive_attr]

    feature_learner = load_feature_learner(args, info_model=info_bool)

    aux_feature_learner = load_feature_learner(args)

    aux_classifier = mlp_classifier(in_dim=args.num_features,
                                    num_classes=sensitive_num_classes,
                                    hidden_dims=[args.num_features*2, args.num_features*2],
                                    drop_rate=args.drop_rate)


    ### adversary: transformer & sensitive attribute decensor
    if args.dataset == 'twitter':
        transformer = mlp_encoder(in_dim=args.num_features,
                                  out_dim=args.num_features,
                                  hidden_dims=[args.num_features*2, args.num_features*2])
    else:
        transformer = mlp_encoder(in_dim=args.num_features,
                                  out_dim=args.num_features,
                                  hidden_dims=[args.num_features*2])


    attr_decensor = mlp_classifier(in_dim=args.num_features,
                               num_classes=sensitive_num_classes,
                               hidden_dims=[args.num_features*2, args.num_features*2],
                               drop_rate=args.adv_drop_rate)
    ## load feature learner
    state_dict = torch.load(os.path.join('../defenses/' + args.dataset +
                                          '_' + args.defense_method, args.defense_model_name + '_feature_model.pth'))
    feature_learner.load_state_dict(state_dict)



    ### use cuda
    if args.use_cuda:
        aux_feature_learner, aux_classifier, transformer, attr_decensor = aux_feature_learner.cuda(), aux_classifier.cuda(), transformer.cuda(), attr_decensor.cuda()
        feature_learner = feature_learner.cuda()

    ### set to train mode
    aux_feature_learner.train()
    aux_classifier.train()
    transformer.train()
    attr_decensor.train()

    ### set feature learner to eval mode (query mode)
    feature_learner.eval()

    ### loss
    if args.surrogate == 'ce':
        criterion = nn.CrossEntropyLoss()
    mseloss = nn.MSELoss()


    logger.info('Epoch \t Pre Train Acc \t Pre Train Loss')


    ### pretrain the auxiliary model
    print('--------pretrain the auxiliary model--------')
    lr_steps = len(train_loader)*args.num_epochs
    opt_aux_feature = torch.optim.Adam(aux_feature_learner.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    opt_aux_classifier = torch.optim.Adam(aux_classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler_aux_feature = torch.optim.lr_scheduler.MultiStepLR(opt_aux_feature, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    scheduler_aux_classifier = torch.optim.lr_scheduler.MultiStepLR(opt_aux_classifier, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, (X, y, s) in enumerate(train_loader):

                if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()

                z_aux = aux_feature_learner(X)
                output = aux_classifier(z_aux)
                loss = criterion(output, s)

                opt_aux_feature.zero_grad()
                opt_aux_classifier.zero_grad()
                loss.backward()
                opt_aux_feature.step()
                opt_aux_classifier.step()

                train_n += s.size(0)
                train_acc += (output.max(1)[1] == s).sum().item()
                train_loss += loss.item()*s.size(0)

                scheduler_aux_feature.step()
                scheduler_aux_classifier.step()

                pbar.update(1)

        logger.info('%d \t %.4f \t %.4f', epoch, train_acc/train_n, train_loss/train_n)


    aux_feature_learner.eval()
    aux_classifier.eval()

    ### evaluate auxiliary model
    test_loss = 0
    test_acc = 0
    test_n = 0
    for i, (X, y, s) in enumerate(test_loader):
        if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
        z_aux = aux_feature_learner(X)
        output = aux_classifier(z_aux)
        loss = criterion(output, s)

        test_n += s.size(0)
        test_acc += (output.max(1)[1] == s).sum().item()
        test_loss += loss.item()*s.size(0)

    logger.info('Auxiliary Test Acc: %.4f \t Auxiliary Test Loss: %.4f', test_acc/test_n, test_loss/test_n)


    ### train the attack model
    logger.info('Epoch \t Transform Loss')
    print('------------pretrain transformer-----------')
    opt_transformer = torch.optim.Adam(transformer.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler_transformer = torch.optim.lr_scheduler.MultiStepLR(opt_transformer, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        start_epoch_time = time.time()
        train_mse_loss = 0
        train_n = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, (X, y, s) in enumerate(train_loader):
                if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
                z = feature_learner(X).detach() ## to infer s from z (attack feature learner)
                z_aux = aux_feature_learner(X).detach()
                z_t = transformer(z)

                if args.dataset == 'twitter':
                    mse_loss = mseloss(z_t, z_aux)
                else:
                    mse_loss = mseloss(z_t, z_aux)*z_aux.size(1)

                ## train transformer to match T(z) with z_aux
                opt_transformer.zero_grad()
                mse_loss.backward()
                opt_transformer.step()

                train_n += s.size(0)
                train_mse_loss += mse_loss.item()*s.size(0)

                scheduler_transformer.step()

                pbar.update(1)

        logger.info('%d \t %.4f', epoch, train_mse_loss/train_n)

    logger.info('Epoch \t Attack Train Acc \t Attack Train Loss \t Transform Loss')
    print('------------train classifier for decensoring-----------')
    opt_decensor = torch.optim.Adam(attr_decensor.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler_decensor = torch.optim.lr_scheduler.MultiStepLR(opt_decensor, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        start_epoch_time = time.time()
        train_loss = 0
        train_mse_loss = 0
        train_acc = 0
        train_n = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, (X, y, s) in enumerate(train_loader):
                if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
                # z = feature_learner(X) ## to infer s from z (attack feature learner)
                # z_aux = aux_feature_learner(X)
                # z_t = transformer(z)
                # mse_loss = mseloss(z_t, z_aux)*z_aux.size(1)
                #
                # ## train transformer to match T(z) with z_aux
                # opt_transformer.zero_grad()
                # mse_loss.backward()
                # opt_transformer.step()

                z = feature_learner(X).detach() ## z is deleted after the above backward operation
                output = attr_decensor(transformer(z))
                loss = criterion(output, s)

                ## train the attack classifier
                # opt_transformer.zero_grad()
                opt_decensor.zero_grad()
                loss.backward()
                # opt_transformer.step()
                opt_decensor.step()

                train_n += s.size(0)
                train_acc += (output.max(1)[1] == s).sum().item()
                train_loss += loss.item()*s.size(0)
                train_mse_loss += mse_loss.item()*s.size(0)

                scheduler_decensor.step()

                pbar.update(1)

        logger.info('%d \t %.4f \t %.4f \t %.4f',
                        epoch, train_acc/train_n, train_loss/train_n, train_mse_loss/train_n)


    ## save models
    torch.save(transformer.state_dict(), os.path.join(args.out_dir, args.model_name + '_transform_model.pth'))
    torch.save(attr_decensor.state_dict(), os.path.join(args.out_dir, args.model_name + '_decensor.pth'))


    ## evaluate the model
    transformer.eval()
    attr_decensor.eval()

    if args.surrogate == 'ce':
        criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_acc = 0
    test_n = 0

    for i, (X, y, s) in enumerate(test_loader):
        if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
        z = feature_learner(X).detach()
        output = attr_decensor(transformer(z))
        loss = criterion(output, s)

        test_n += s.size(0)
        test_acc += (output.max(1)[1] == s).sum().item()
        test_loss += loss.item()*s.size(0)

    print('Adv Test Acc: %.4f \t Adv Test Loss: %.4f'%(test_acc/test_n, test_loss/test_n))
    logger.info('Adv Test Acc: %.4f \t Adv Test Loss: %.4f', test_acc/test_n, test_loss/test_n)




if __name__ == '__main__':
    args = get_args()

    args.out_dir = args.dataset + '_' + args.out_dir

    ### log information
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    assert (args.defense_model_name is not None)
    assert args.sensitive_attr is not 'none'

    print('sensitive variable: {}'.format(args.sensitive_attr))

    args.model_name = 'defense_{}_defense_params_{}_attacker_budget_{}_attack_params_batch_{}_lr_{}_decay{}_drop_{}' \
                       .format(args.defense_method,
                               args.defense_model_name,
                               args.attacker_size,
                               args.batch_size,
                               args.lr,
                               args.weight_decay,
                               args.adv_drop_rate)

    # if args.test:
    #     logfile = os.path.join(args.out_dir, args.model_name + '_test_output.log')
    # else:
    logfile = os.path.join(args.out_dir, args.model_name + '_output.log')

    if os.path.exists(logfile):
        os.remove(logfile)

    logging.basicConfig(format='[%(asctime)s] - %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO,
                        filename=logfile)
    logger.info(args)

    ### random seeds
    torch.backends.cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    main(args)
