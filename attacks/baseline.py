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
from data_preprocess.config import dataset_class_params

## utils
from utils import normal_weight_init, print_params_names

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

    parser.add_argument('--drop-rate', type=float, default=0.0, help='only for loading the feature learner')
    parser.add_argument('--adv-drop-rate', type=float, default=0.0, help='dropout rate to train the adversary model')

    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--out-dir', default='baseline_attack', type=str)
    parser.add_argument('--defense-method', default='adv_censor', type=str)
    parser.add_argument('--defense-model-name', default=None, type=str)
    parser.add_argument('--model-name', default=None, type=str)

    parser.add_argument('--seed', default=1, type=int)

    return parser.parse_args()


def main(args):
    info_bool = ('info' in args.defense_method) or ('vae' in args.defense_method)
    if 'no_rand' in args.defense_method: info_bool = False
    print('Is info model or not? {}'.format(info_bool))

    train_loader, test_loader = load_attack_data(args)

    sensitive_num_classes = dataset_class_params[args.dataset][args.sensitive_attr]

    feature_learner = load_feature_learner(args, info_model=info_bool)

    ## create adversary classifier (learn on the features)
    adversary = mlp_classifier(in_dim=args.num_features,
                               num_classes=sensitive_num_classes,
                               hidden_dims=[args.num_features*2, args.num_features*2],
                               drop_rate=args.adv_drop_rate)


    ## load feature learner
    if args.defense_method != 'none':
        state_dict = torch.load(os.path.join('../defenses/' + args.dataset +
                                              '_' + args.defense_method, args.defense_model_name + '_feature_model.pth'))
        feature_learner.load_state_dict(state_dict)


    if args.use_cuda:
        feature_learner, adversary = feature_learner.cuda(), adversary.cuda()

    ### set modes
    feature_learner.eval() ## query mode
    adversary.train()

    lr_steps = len(train_loader)*args.num_epochs
    opt_adversary = torch.optim.Adam(adversary.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler_adversary = torch.optim.lr_scheduler.MultiStepLR(opt_adversary, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    # scheduler_adversary = torch.optim.lr_scheduler.ExponentialLR(opt_adversary, gamma=0.95)
    if args.surrogate == 'ce':
        criterion = nn.CrossEntropyLoss()

    logger.info('Epoch \t Adv Train Acc \t Adv Train Loss')
    ### start training
    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        start_epoch_time = time.time()
        train_adv_loss = 0
        train_adv_acc = 0
        train_n = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, (X, y, s) in enumerate(train_loader):
                if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
                z = feature_learner(X).detach()

                ## train normal classifier
                adv_output = adversary(z)
                adv_loss = criterion(adv_output, s)

                opt_adversary.zero_grad()
                adv_loss.backward()
                opt_adversary.step()

                ### record some training info
                train_n += s.size(0)
                train_adv_acc += (adv_output.max(1)[1] == s).sum().item()
                train_adv_loss += adv_loss.item()*s.size(0)

                scheduler_adversary.step()

                pbar.update(1)

        print('%d training acc: %.4f training loss: %.4f'%(epoch, train_adv_acc/train_n, train_adv_loss/train_n), flush=True)
        logger.info('%d \t %.4f \t %.4f', epoch, train_adv_acc/train_n, train_adv_loss/train_n)


    ### evaluation
    adversary.eval()
    test_adv_loss = 0
    test_adv_acc = 0
    test_n = 0
    for i, (X, y, s) in enumerate(test_loader):
        if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()
        z = feature_learner(X).detach()
        adv_output = adversary(z)
        adv_loss = criterion(adv_output, s)

        test_n += y.size(0)
        test_adv_acc += (adv_output.max(1)[1] == s).sum().item()
        test_adv_loss += adv_loss.item()*y.size(0)

    print('Adv Test Acc: %.4f Adv Test Loss: %.4f'%(
                test_adv_acc/test_n, test_adv_loss/test_n), flush=True)
    logger.info('Adv Test Acc: %.4f \t Adv Test Loss: %.4f',
                test_adv_acc/test_n, test_adv_loss/test_n)

    torch.save(adversary.state_dict(), os.path.join(args.out_dir, args.model_name+'.pth'))



if __name__ == '__main__':
    args = get_args()

    args.out_dir = args.dataset + '_' + args.out_dir

    ### log information
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    assert (args.defense_model_name != None)
    assert args.sensitive_attr != 'none'

    print('sensitive variable: {}'.format(args.sensitive_attr))
    args.model_name = 'defense_{}_defense_params_{}_attacker_budget_{}_attack_params_batch_{}_lr_{}_decay_{}_drop_{}' \
                       .format(args.defense_method,
                               args.defense_model_name,
                               args.attacker_size,
                               args.batch_size,
                               args.lr,
                               args.weight_decay,
                               args.adv_drop_rate)


    logfile = os.path.join(args.out_dir, args.model_name +'_output.log')
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
