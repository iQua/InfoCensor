import torch
import torch.nn as nn
import torch.nn.functional as F ## F.softmax, F.log_softmax
import torch.nn.init as init
import math
import numpy as np
import gmm

from models.encoder import *
from models.decoder import *

## dataset
from data_preprocess.load_health import load_health
from data_preprocess.load_utkface import load_utkface
from data_preprocess.load_nlps import load_twitter
from data_preprocess.load_german import load_german


def one_hot_encoding(label, num_classes, use_cuda=False):
    label_cpu = label.cpu().data
    onehot = torch.zeros(label_cpu.size(0), num_classes).scatter_(1, label_cpu.unsqueeze(1), 1.).float()
    return onehot.cuda() if use_cuda else onehot

def pairwise_kl(mu, sigma, add_third_term=False):
    # k = K.shape(sigma)[1]
    # k = K.cast(k, 'float32')
    d = float(sigma.size(1))

    # var = K.square(sigma) + 1e-8
    var = sigma**2 + 1e-8

    # var_inv = K.tf.reciprocal(var)
    # var_diff = K.dot(var, K.transpose(var_inv))
    var_inv = 1./var
    var_diff = torch.matmul(var, torch.t(var_inv)) ## tr(S2^-1 S1)

    # r = K.dot(mu * mu, K.transpose(var_inv))
    # r2 = K.sum(mu * mu * var_inv, axis=1)
    # mu_var_mu = 2 * K.dot(mu, K.transpose(mu * var_inv))
    # mu_var_mu = r - mu_var_mu + K.transpose(r2)

    r = torch.matmul(mu**2, torch.t(var_inv)) ## batch x batch
    r2 = torch.sum(mu*mu*var_inv, dim=1, keepdim=True) ## batch x 1
    mu_var_mu = 2 * torch.matmul(mu, torch.t(mu * var_inv))
    mu_var_mu = r - mu_var_mu + torch.t(r2)

    if add_third_term:  # this term cancels out for examples in a batch (meaning = 0)
        log_det = torch.sum(torch.log(var), dim=1, keepdim=True)
        log_def_diff = log_det - torch.t(log_det)
    else:
        log_det_diff = 0.

    KL = 0.5 * (var_diff + mu_var_mu + log_det_diff - d) ## batch x batch
    return KL


def kl_conditional_and_marg(mu, sigma):
    b = float(sigma.size(0))
    d = float(sigma.size(1))

    ### H(z|x)
    H = 0.5*(torch.sum(torch.log(sigma**2 + 1e-8), dim=1)
                                        + d*(1 + math.log(2 * math.pi))) ## d/2*log(2e\pi det)

    KL = pairwise_kl(mu, sigma)


    return 1.0/b * torch.mean(torch.sum(KL, dim=1) + (b - 1) * H - math.log(b))

def information_bottleneck(mu, sigma):
    return F.relu(torch.mean(pairwise_kl(mu, sigma))) ## to guarantee the estimation is larger than 0

def conditional_gaussian_entropy(mu, sigma):
    d = float(sigma.size(1))
    H = 0.5*(torch.sum(torch.log(sigma**2 + 1e-8), dim=1)
                                         + d*(1 + math.log(2 * math.pi)))
    return torch.mean(H)


def variational_mutual_information_estimator(mu, sigma, s, sensitive_num_classes, iters=10):
    KL = F.relu(pairwise_kl(mu, sigma))
    KL_exp = torch.exp(-KL)
    N = float(s.size(0))

    mutual_info = 0

    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)

        if total_slabel > 0:
            slabel_mask_matrix = slabel_mask.repeat(KL_exp.size(0), 1)
            phi = slabel_mask_matrix/N/float(total_slabel)

            for iter in range(iters):
                psi = 1./N*phi.t()/(torch.sum(phi, dim=1) + 1e-16)
                inter_mat = psi*slabel_mask_matrix.t()*KL_exp
                phi = (1./float(total_slabel)*inter_mat.t())/(torch.sum(inter_mat, dim=1) + 1e-16)

            phi, psi = phi.detach(), psi.detach()

            mutual_info += torch.sum(phi.t()*slabel_mask_matrix.t()*KL)

    return mutual_info/float(sensitive_num_classes)


def mcmc_mutual_information_estimator(mu, sigma, s, sensitive_num_classes, n=50000):

    sigma = np.square(sigma)
    weights = np.ones(mu.shape[0])/mu.shape[0]
    data_gmm = gmm.GMM(mu, sigma, weights)

    mutual_info = 0
    for slabel in range(sensitive_num_classes):
        means, covars = [], []
        counter = 0
        for i in range(s.shape[0]):
            if s[i] == slabel:
                means.append(mu[i])
                covars.append(sigma[i])
                counter += 1
        means, covars = np.array(means), np.array(covars)
        weights = np.ones(means.shape[0])/means.shape[0]
        s_data_gmm = gmm.GMM(means, covars, weights)

        samples = s_data_gmm.sample(n)
        mutual_info += 1./n * np.sum(s_data_gmm.log_likelihood(samples) - data_gmm.log_likelihood(samples)) * counter

    return mutual_info/float(s.shape[0])


def batch_diag(a):
    b = torch.eye(a.size(1), device=a.device)
    return a.unsqueeze(2).expand(*a.size(), a.size(1))*b


def gaussian_mutual_information_estimator(mu, sigma, s, sensitive_num_classes):
    d = float(mu.size(1))
    mu_1 = torch.mean(mu, dim=0)
    mu_d = mu-mu_1.repeat(mu.size(0), 1)
    Sigma = batch_diag(sigma) + torch.bmm(mu_d.unsqueeze(2), mu_d.unsqueeze(1))
    sigma_1 = torch.mean(Sigma, dim=0)

    mutual_info = 0
    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)
        mu_2 = torch.sum(slabel_mask*mu, dim=0)/(total_slabel+1e-8)
        sigma_2 = torch.sum(slabel_mask*Sigma, dim=0)/(total_slabel+1e-8)

        sigma_2_inverse = torch.inverse(sigma_1)

        mutual_info += 0.5*(torch.log(F.relu(sigma_2.det()/(sigma_1.det()+1e-8))+1e-8) - d +
                            torch.trace(sigma_2_inverse * sigma_1) +
                            torch.mm(torch.mm((mu_2 - mu_1).unsqueeze(0), sigma_2_inverse),
                            (mu_2 - mu_1).unsqueeze(1)))

    return mutual_info


def fair_pred_mutual_information(output, s, sensitive_num_classes, T=0.1):

    mutual_info = 0
    gumbel_pred = F.gumbel_softmax(output, tau=T)
    class_prob = torch.sum(gumbel_pred, dim=0)/torch.sum(gumbel_pred)

    for slabel in range(sensitive_num_classes):
        slabel_mask = torch.eq(s, slabel).float()
        total_slabel = torch.sum(slabel_mask)
        slabel_pred = slabel_mask.unsqueeze(1)*gumbel_pred
        cond_prob = torch.sum(slabel_pred, dim=0)/(torch.sum(slabel_pred)+ 1e-8)

        mutual_info += torch.sum(cond_prob*torch.log(cond_prob/(class_prob + 1e-8) + 1e-8))

    return mutual_info/float(sensitive_num_classes)


def fairness_metric(sensitive_class_count, sensitive_num_classes):

    sensitive_total_count = [0]*len(sensitive_class_count[0])
    sensitive_class_freq = {}
    for slabel in range(sensitive_num_classes):
        sensitive_class_freq[slabel] = np.array(sensitive_class_count[slabel])/sum(sensitive_class_count[slabel])
        sensitive_total_count = [sensitive_class_count[slabel][i]+sensitive_total_count[i] for i in range(len(sensitive_total_count))]

    sensitive_total_freq = np.array(sensitive_total_count)/sum(sensitive_total_count)

    mutual_info = 0
    for slabel in range(sensitive_num_classes):
        mutual_info += np.sum(sensitive_class_freq[slabel] * np.log(sensitive_class_freq[slabel]/(sensitive_total_freq+1e-8)+1e-8))

    return sensitive_class_freq, mutual_info/float(sensitive_num_classes)

def spd_metric(sensitive_class_freq, sensitive_num_classes):
    total_spd = 0
    max_spd = 0
    count = 0

    sensitive_class_freq_list = []
    for slabel in range(sensitive_num_classes):
        sensitive_class_freq_list.append(sensitive_class_freq[slabel])
    # sensitive_class_freq_array = np.concatenate(sensitive_class_freq_list, axis=0)

    for i in range(sensitive_num_classes):
        for j in range(i+1, sensitive_num_classes):
            abs_spd_arr = np.abs(sensitive_class_freq_list[i] - sensitive_class_freq_list[j])
            if np.max(abs_spd_arr) > max_spd:
                max_spd = np.max(abs_spd_arr)
            total_spd += np.sum(abs_spd_arr)
            count += abs_spd_arr.shape[0]

    return max_spd, total_spd/count



def print_params_names(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, '\n', param.data.size(), '\n', param.data)


def load_data_model(args, info_model=False, return_decoder=False):

    print('randomize the representations or not: {}'.format(info_model))

    if args.dataset == 'german':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)

        print('input dimension: {} \t target classes: {} \t sensitive classes: {}'.format(input_dim, target_num_classes, sensitive_num_classes), flush=True)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model) ## learn q(z|x)

        if return_decoder:
            feature_decoder = mlp_decoder(in_dim=args.num_features+sensitive_num_classes, out_dim=input_dim, drop_rate=args.drop_rate)

    if args.dataset == 'health':
        train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_health(attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        binarize=False,
                                                                                                        batch_size=args.batch_size)

        print('input dimension: {} \t target classes: {} \t sensitive classes: {}'.format(input_dim, target_num_classes, sensitive_num_classes), flush=True)

        ### create models
        feature_learner = mlp_encoder(in_dim=input_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model) ## learn q(z|x)

        if return_decoder:
            feature_decoder = mlp_decoder(in_dim=args.num_features+sensitive_num_classes, out_dim=input_dim, drop_rate=args.drop_rate)

    elif args.dataset == 'utkface':
        train_loader, test_loader, img_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr=args.target_attr,
                                                                                                        sensitive_attr=args.sensitive_attr,
                                                                                                        random_seed=args.seed,
                                                                                                        batch_size=args.batch_size)
        print('image dimension: {} \t target classes: {} \t sensitive classes: {}'.format(img_dim, target_num_classes, sensitive_num_classes), flush=True)

        feature_learner = lenet_encoder(img_dim=img_dim, out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

        if return_decoder:
            feature_decoder = lenet_decoder(input_dim=args.num_features+sensitive_num_classes, img_dim=img_dim, drop_rate=args.drop_rate)

    elif args.dataset == 'twitter':
        train_loader, test_loader, voc_size, target_num_classes, sensitive_num_classes = load_twitter(sensitive_attr=args.sensitive_attr,
                                                                                                      random_seed=args.seed,
                                                                                                      batch_size=args.batch_size)

        print('vocabulary size: {} \t target classes: {} \t sensitive classes: {}'.format(voc_size, target_num_classes, sensitive_num_classes), flush=True)

        feature_learner = lstm_encoder(voc_size=voc_size, embedding_dim=32,
                                       out_dim=args.num_features, drop_rate=args.drop_rate, info_model=info_model)

        if return_decoder:
            feature_decoder = lstm_decoder(voc_size=voc_size, fea_dim=args.num_features+sensitive_num_classes,
                                       hidden_dim=args.num_features, drop_rate=args.drop_rate)


    if return_decoder:
        return train_loader, test_loader, feature_learner, feature_decoder
    else:
        return train_loader, test_loader, feature_learner


def normal_weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(normal_weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def check_args(args):
    if args.dataset == 'health':
        assert args.target_attr == 'charlson'
        assert (args.sensitive_attr == 'gender' or args.sensitive_attr == 'age')
    elif args.dataset == 'twitter':
        assert args.target_attr == 'age'
        assert (args.sensitive_attr == 'gender' or args.sensitive_attr == 'identity')
    elif args.dataset == 'utkface':
        assert args.target_attr == 'age'
        assert (args.sensitive_attr == 'gender' or args.sensitive_attr == 'race')
    elif args.dataset == 'german':
        assert args.target_attr == 'credit'
        assert (args.sensitive_attr == 'gender' or args.sensitive_attr == 'age')
