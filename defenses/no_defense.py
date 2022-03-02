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
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
## models
sys.path.insert(1, '../')
from data_preprocess.config import dataset_class_params
from models.classifier import mlp_classifier


from utils import load_data_model, check_args

from eval_utils import evaluate, store_representations



import argparse
if torch.cuda.is_available():
    import setGPU

logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)

    parser.add_argument('--dataset', default='health', type=str, choices=['german', 'health', 'utkface', 'twitter'])
    parser.add_argument('--num-epochs', default=50, type=int)

    parser.add_argument('--num-features', default=128, type=int)
    parser.add_argument('--target-attr', default='age', type=str, choices=['age', 'charlson', 'credit'])
    parser.add_argument('--sensitive-attr', default='none', type=str, choices=['none', 'age', 'gender', 'race', 'identity'])
    parser.add_argument('--surrogate', default='ce', type=str)


    ## follow the existing work to train gan, we use adam optimizer
    parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
    parser.add_argument("--beta1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--drop-rate', type=float, default=0.1, help='dropout rate to train feature learner')

    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--out-dir', default='no_defense', type=str)

    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()


def main(args):

    train_loader, test_loader, feature_learner = load_data_model(args, info_model=False)

    target_num_classes = int(dataset_class_params[args.dataset][args.target_attr])

    normal_classifier = mlp_classifier(in_dim=args.num_features, num_classes=target_num_classes, drop_rate=args.drop_rate)

    ### use cuda
    if args.use_cuda:
        feature_learner, normal_classifier = feature_learner.cuda(), normal_classifier.cuda()

    ### set to train mode
    feature_learner.train()
    normal_classifier.train()

    ### create optimizers
    lr_steps = len(train_loader)*args.num_epochs
    opt_feature = torch.optim.Adam(feature_learner.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    opt_classifier = torch.optim.Adam(normal_classifier.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    scheduler_feature = torch.optim.lr_scheduler.MultiStepLR(opt_feature, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
    scheduler_classifier = torch.optim.lr_scheduler.MultiStepLR(opt_classifier, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    ## loss
    criterion = nn.CrossEntropyLoss()

    ### start training
    logger.info('Epoch \t Normal Train Acc \t Normal Train Loss')
    batch_size = args.batch_size
    for epoch in range(args.num_epochs):
        print('training epoch {}'.format(epoch), flush=True)
        start_epoch_time = time.time()
        train_normal_loss = 0
        train_normal_acc = 0
        train_n = 0
        with tqdm(total=len(train_loader)) as pbar:
            for i, (X, y, s) in enumerate(train_loader):
                if args.use_cuda: X, y, s = X.cuda(), y.cuda(), s.cuda()

                z = feature_learner(X)
                pred = normal_classifier(z)
                celoss = criterion(pred, y)

                ## only train the model on the normal task
                opt_classifier.zero_grad()
                opt_feature.zero_grad()
                celoss.backward()
                opt_feature.step()
                opt_classifier.step()


                ## record metrics
                train_n += y.size(0)
                train_normal_acc += (pred.max(1)[1] == y).sum().item()
                train_normal_loss += celoss.item()*y.size(0)

                scheduler_feature.step()
                scheduler_classifier.step()

                pbar.update(1)

        logger.info('%d \t\t %.4f \t\t %.4f', epoch, train_normal_acc/train_n, train_normal_loss/train_n)


    ### save models
    torch.save(feature_learner.state_dict(), os.path.join(args.out_dir, args.model_name + '_feature_model.pth'))
    torch.save(normal_classifier.state_dict(), os.path.join(args.out_dir, args.model_name + '_classifier_model.pth'))


if __name__ == '__main__':
    args = get_args()
    check_args(args)

    args.out_dir = args.dataset + '_' + args.out_dir

    args.model_name = 'target_{}_sensitive_{}_num_features_{}'.format(args.target_attr,
                                                                      args.sensitive_attr,
                                                                      args.num_features)

    ### log information
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    if args.test:
        logfile = os.path.join(args.out_dir, args.model_name + '_test_output.log')
    else:
        logfile = os.path.join(args.out_dir, args.model_name + '_train_output.log')
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

    if args.test:
        evaluate(args, info_model=False)
        if args.dataset == 'utkface':
            store_representations(args, info_model=False)
    else:
        main(args)
