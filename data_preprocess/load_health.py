import numpy as np
import pandas as pd
import torch
# import setGPU

from sklearn.model_selection import train_test_split

DATA_DIR = '../health_data/'
HEALTH_PATH = DATA_DIR + 'health.csv'


def create_health_dataset(attr='age', binarize=True):
    d = pd.read_csv(HEALTH_PATH)
    d = d[d['YEAR_t'] == 'Y3']
    sex = d['sexMISS'] == 0
    age = d['age_MISS'] == 0
    d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    d = d[sex & age]

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    ages = d[['age_%d5' % (i) for i in range(0, 9)]]
    sexs = d[['sexMALE', 'sexFEMALE']]
    charlson = d['CharlsonIndexI_max']

    x = d.drop(
        ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1).values

    labels = gather_labels(x)
    xs = np.zeros_like(x)
    for i in range(len(labels)):
        xs[:, i] = x[:, i] > labels[i]

    col_indices = np.nonzero(np.mean(xs, axis=0) > 0.05)[0]
    x = x[:, col_indices]
    if binarize:
        x = xs[:, col_indices].astype(np.float32)
    else:
        x = (x - np.min(x, axis=0)) / np.max(x, axis=0)
        # mn = np.mean(x, axis=0)
        # std = np.std(x, axis=0)
        # x = whiten(x, mn, std)

    u = sexs.values[:, 0]
    v = np.argmax(ages.values, axis=1)
    a = u if attr == 'gender' else v

    y = (charlson.values > 0).astype(np.int64)
    return x, y, a

def create_health_dataset_full(binarize=True):
    d = pd.read_csv(HEALTH_PATH)
    d = d[d['YEAR_t'] == 'Y3']
    sex = d['sexMISS'] == 0
    age = d['age_MISS'] == 0
    d = d.drop(['DaysInHospital', 'MemberID_t', 'YEAR_t'], axis=1)
    d = d[sex & age]

    def gather_labels(df):
        labels = []
        for j in range(df.shape[1]):
            if type(df[0, j]) is str:
                labels.append(np.unique(df[:, j]).tolist())
            else:
                labels.append(np.median(df[:, j]))
        return labels

    ages = d[['age_%d5' % (i) for i in range(0, 9)]]
    sexs = d[['sexMALE', 'sexFEMALE']]
    charlson = d['CharlsonIndexI_max']

    x = d.drop(
        ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE', 'CharlsonIndexI_max', 'CharlsonIndexI_min',
                                                  'CharlsonIndexI_ave', 'CharlsonIndexI_range', 'CharlsonIndexI_stdev',
                                                  'trainset'], axis=1).values

    labels = gather_labels(x)
    xs = np.zeros_like(x)
    for i in range(len(labels)):
        xs[:, i] = x[:, i] > labels[i]

    col_indices = np.nonzero(np.mean(xs, axis=0) > 0.05)[0]
    x = x[:, col_indices]
    if binarize:
        x = xs[:, col_indices].astype(np.float32)
    else:
        x = (x - np.min(x, axis=0)) / np.max(x, axis=0)
        # mn = np.mean(x, axis=0)
        # std = np.std(x, axis=0)
        # x = whiten(x, mn, std)

    u = sexs.values[:, 0]
    v = np.argmax(ages.values, axis=1)

    y = (charlson.values > 0).astype(np.int64)
    return x, y, u, v


def load_health(attr='age', train_size=0.8, random_seed=0,
                binarize=True, batch_size=128):
    X, y, s = create_health_dataset(attr, binarize)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s,
                                                                         train_size=train_size,
                                                                         random_state=random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder(X=X_train, y=y_train, s=s_train),
        batch_size=batch_size, shuffle=True,
        num_workers=2,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(X=X_test, y=y_test,
        s=s_test, is_training=False),
        batch_size=batch_size, shuffle=False,
        num_workers=2)

    ## return train DataLoader, test DataLoader, number of features,
    return train_loader, test_loader, X_train.shape[1], y_train.max()+1, s_train.max()+1


def load_health_attack(attr='age', train_size=0.8, attacker_size=0.5, random_seed=0,
                binarize=True, batch_size=128):
    X, y, s = create_health_dataset(attr, binarize)
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s,
                                                                         train_size=train_size,
                                                                         random_state=random_seed)

    X_train, _, y_train, _, s_train, _ = train_test_split(X_train, y_train, s_train,
                                                                         train_size=attacker_size,
                                                                         random_state=random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder(X=X_train, y=y_train, s=s_train),
        batch_size=batch_size, shuffle=True,
        num_workers=2,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder(X=X_test, y=y_test,
        s=s_test, is_training=False),
        batch_size=batch_size, shuffle=False,
        num_workers=2)

    ## return train DataLoader, test DataLoader, number of features,
    return train_loader, test_loader, X_train.shape[1], y_train.max()+1, s_train.max()+1



def load_health_full(train_size=0.8, random_seed=0,
                    binarize=True, batch_size=128):
    X, y, s1, s2 = create_health_dataset_full(binarize)
    X_train, X_test, y_train, y_test, s1_train, s1_test, s2_train, s2_test = train_test_split(X, y, s1, s2,
                                                                                             train_size=train_size,
                                                                                             random_state=random_seed)

    train_loader = torch.utils.data.DataLoader(
        dataset=Feeder_full(X=X_train, y=y_train, s1=s1_train, s2=s2_train),
        batch_size=batch_size, shuffle=True,
        num_workers=2,
        drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=Feeder_full(X=X_test, y=y_test, s1=s1_test, s2=s2_test, is_training=False),
        batch_size=batch_size, shuffle=False,
        num_workers=2)

    ## return train DataLoader, test DataLoader, number of features,
    return train_loader, test_loader, X_train.shape[1], y_train.max()+1, s1_train.max()+1, s2_train.max()+1

class Feeder(torch.utils.data.Dataset):

    def __init__(self,
                 X, y, s,
                 is_training = True,
                 normalization=True
                 ):

        self.X, self.y, self.s = X.astype(np.float32), y, s
        self.is_training = is_training


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data, target, slabel = self.X[index], self.y[index], self.s[index]
        return data, target, slabel


class Feeder_full(torch.utils.data.Dataset):

    def __init__(self,
                 X, y, s1, s2,
                 is_training = True,
                 normalization=True
                 ):

        self.X, self.y, self.s1, self.s2 = X.astype(np.float32), y, s1, s2
        self.is_training = is_training


    def __len__(self):
        return self.X.shape[0]

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data, target, slabel1, slabel2 = self.X[index], self.y[index], self.s1[index], self.s2[index]
        return data, target, slabel1, slabel2



if __name__ == '__main__':
    train_loader, test_loader, _, _, _ = load_health(attr='age', binarize=False, batch_size=128)
    print(len(train_loader))
