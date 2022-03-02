import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
DATA_DIR = '../german_data/'
DATA_PATH = DATA_DIR + 'german_credit.csv'

def load_german(attr='age', batch_size=128, train_size=0.8, random_seed=0):
    data = pd.read_csv(DATA_PATH)

    y = (data['Class']=='Good').values.astype(np.int64)
    a = (data['Age'] > 30).values.astype(np.int64)
    g = data['Personal.Male.Divorced.Seperated'].values.astype(np.int64) + \
        data['Personal.Male.Married.Widowed'].values.astype(np.int64) + \
        data['Personal.Male.Single'].values.astype(np.int64)

    s = a if attr == 'age' else g
    X = data.drop(['Personal.Male.Divorced.Seperated', 'Personal.Male.Divorced.Seperated', 'Personal.Male.Divorced.Seperated', 'Personal.Female.NotSingle',
                                              'Personal.Female.Single', 'Class'], axis=1).values
    X = X.astype(np.float32)

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

    return train_loader, test_loader, X_train.shape[1], y_train.max()+1, s_train.max()+1


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

if __name__ == '__main__':
    train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_german()
    print(input_dim, target_num_classes, sensitive_num_classes)
    
