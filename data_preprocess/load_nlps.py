import pickle
import os, sys
import pandas as pd
from tqdm import tqdm
import torch
from datetime import datetime
from torch.utils import data
from torchtext import datasets
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence as pad_sequence
from sklearn.model_selection import train_test_split

from .twitter_utils import normalize_text

ROOT_DATA_PATH = '/opt/data/zth/nlp_dataset'
# ROOT_DATA_PATH = './data'
### remove the sentence with length < 20 and remove the id with number of tweets < 50
def preprocess_twitter(min_len=20, min_tweets=500):
    data = pd.read_csv(os.path.join(ROOT_DATA_PATH, 'author-profiling.tsv'),
                     sep='\t',
                     names=['index', 'id', 'gender', 'age', 'text'],
                     skiprows=1, index_col=False, lineterminator='\n',
                     encoding='utf8')
    arr = []
    ids = []
    ages = []
    genders = []
    for ind in tqdm(range(len(data))):
        try:
            t = normalize_text(data.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            arr.append(t)
            ids.append(data.iloc[ind].id)
            ages.append(data.iloc[ind].age)
            genders.append(0 if data.iloc[ind].gender == 'male' else 1)
        except:
            pass

    ### remove the id with number of tweets < min_tweets
    id_dic = Counter(ids)
    ids_to_del = []
    for id in id_dic:
        if id_dic[id] < min_tweets: ids_to_del.append(id)

    for id in ids_to_del: del id_dic[id]

    ## update id dictionary
    id_counter = 0
    for id in id_dic:
        id_dic[id] = (id_counter, id_dic[id])
        id_counter += 1

    ### update vocab
    counter = Counter()
    for tokens in arr:
        counter.update(tokens)
    vocab = Vocab(counter, min_freq=1)

    ### params
    voc_size = len(vocab)
    num_ids = len(id_dic)
    num_ages = 5
    num_genders = 2

    voc_to_idx_arr = [[vocab[token] for token in tokens] for tokens in arr]

    dataX, dataY, dataI, dataG = [], [], [], []

    for i in range(len(ids)):
        if ids[i] in id_dic:
            dataX.append(voc_to_idx_arr[i])
            dataY.append(ages[i])
            dataI.append(id_dic[ids[i]][0])
            dataG.append(genders[i])

    data = (dataX, dataY, dataI, dataG)

    ## save data_file
    data_file_path = os.path.join(ROOT_DATA_PATH, 'pan16_full.pkl')
    if not os.path.exists(data_file_path):
        with open(data_file_path, 'wb') as fid:
            pickle.dump((data, voc_size, num_ages, num_ids, num_genders), fid)
    return (data, voc_size, num_ages, num_ids, num_genders)


def getvocab(min_len=20, min_tweets=500):
    data = pd.read_csv(os.path.join(ROOT_DATA_PATH, 'author-profiling.tsv'),
                     sep='\t',
                     names=['index', 'id', 'gender', 'age', 'text'],
                     skiprows=1, index_col=False, lineterminator='\n',
                     encoding='utf8')
    arr = []
    ids = []
    ages = []
    genders = []
    for ind in tqdm(range(len(data))):
        try:
            t = normalize_text(data.iloc[ind].text)
            if len(t) < min_len:
                continue
            if len(set(t)) == 1 and t[0] == MENTION: continue
            arr.append(t)
            ids.append(data.iloc[ind].id)
            ages.append(data.iloc[ind].age)
            genders.append(0 if data.iloc[ind].gender == 'male' else 1)
        except:
            pass

    ### remove the id with number of tweets < min_tweets
    id_dic = Counter(ids)
    ids_to_del = []
    for id in id_dic:
        if id_dic[id] < min_tweets: ids_to_del.append(id)

    for id in ids_to_del: del id_dic[id]

    ## update id dictionary
    id_counter = 0
    for id in id_dic:
        id_dic[id] = (id_counter, id_dic[id])
        id_counter += 1

    ### update vocab
    counter = Counter()
    for tokens in arr:
        counter.update(tokens)
    vocab = Vocab(counter, min_freq=1)

    return vocab


class TwitterData(data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        text, y, id, g = self.data[0][index], self.data[1][index], self.data[2][index], self.data[3][index]
        return text, y, id, g

    def __len__(self):
        return len(self.data[0])

def twitter_collate_batch_gender(batch):
    text_list, y_list, s_list = [], [], []

    for (text, y, _, s) in batch:
         text = torch.tensor(text, dtype=torch.int64)
         text_list.append(text)
         y_list.append(y)
         s_list.append(s)

    y_list = torch.tensor(y_list, dtype=torch.int64)
    s_list = torch.tensor(s_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=1)

    return text_list, y_list, s_list


def twitter_collate_batch_identity(batch):
    text_list, y_list, s_list = [], [], []

    for (text, y, s, _) in batch:
         text = torch.tensor(text, dtype=torch.int64)
         text_list.append(text)
         y_list.append(y)
         s_list.append(s)

    y_list = torch.tensor(y_list, dtype=torch.int64)
    s_list = torch.tensor(s_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=1)

    return text_list, y_list, s_list

def preprocess_yelp_dataset(dataset, language='basic_english'):
    tokenizer = get_tokenizer(language)
    counter = Counter()
    for (label, line) in dataset:   ## the structure is (label, text)
        counter.update(tokenizer(line))

    vocab = Vocab(counter, min_freq=1)

    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1

    return text_pipeline, label_pipeline



def yelp_collate_batch(batch):
    text_list, label_list = [], []
    for (label, text) in batch:
         text = torch.tensor(text, dtype=torch.int64)
         text_list.append(text)
         label_list.append(label)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=1)

    return text_list, label_list


def load_twitter(root=ROOT_DATA_PATH, train_size=0.8, batch_size=128, random_seed=0, sensitive_attr='identity'):

    try:
        data_file_path = os.path.join(root, 'pan16_full.pkl')
        print('try to load preprocessed data from {}'.format(data_file_path))
        with open(data_file_path, 'rb') as fid:
            twitterdata, voc_size, num_ages, num_ids, num_genders = pickle.load(fid)
    except:
        print('start to prepreprocess')
        twitterdata, voc_size, num_ages, num_ids, num_genders = preprocess_twitter()

    X, y, id, g = twitterdata[0], twitterdata[1], twitterdata[2], twitterdata[3]

    X_train, X_test, y_train, y_test, id_train, id_test, g_train, g_test = train_test_split(X, y, id, g,
                                                                                             train_size=train_size,
                                                                                             random_state=random_seed)
    # (s_train, s_test, num_sensitive_classes) = (id_train, id_test, num_ids) \
    #             if sensitive_attr == 'identity' else (g_train, g_test, num_genders)

    train_dataset, test_dataset = TwitterData((X_train, y_train, id_train, g_train)), \
                                    TwitterData((X_test, y_test, id_test, g_test))

    if sensitive_attr == 'identity':
        num_sensitive_classes = num_ids
        twitter_collate_batch = twitter_collate_batch_identity
    elif sensitive_attr == 'gender':
        twitter_collate_batch = twitter_collate_batch_gender
        num_sensitive_classes = num_genders


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True,
                                   collate_fn=twitter_collate_batch)

    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False,
                                   collate_fn=twitter_collate_batch)
    return train_loader, test_loader, voc_size, num_ages, num_sensitive_classes


def load_twitter_attack(root=ROOT_DATA_PATH, train_size=0.8, attacker_size=0.5, batch_size=128, random_seed=0, sensitive_attr='identity'):

    try:
        data_file_path = os.path.join(root, 'pan16_full.pkl')
        print('try to load preprocessed data from {}'.format(data_file_path))
        with open(data_file_path, 'rb') as fid:
            twitterdata, voc_size, num_ages, num_ids, num_genders = pickle.load(fid)
    except:
        print('start to prepreprocess')
        twitterdata, voc_size, num_ages, num_ids, num_genders = preprocess_twitter()

    X, y, id, g = twitterdata[0], twitterdata[1], twitterdata[2], twitterdata[3]

    X_train, X_test, y_train, y_test, id_train, id_test, g_train, g_test = train_test_split(X, y, id, g,
                                                                                             train_size=train_size,
                                                                                             random_state=random_seed)
    X_train, _, y_train, _, id_train, _, g_train, _ = train_test_split(X_train, y_train, id_train, g_train,
                                                                                             train_size=attacker_size,
                                                                                             random_state=random_seed)
    train_dataset, test_dataset = TwitterData((X_train, y_train, id_train, g_train)), \
                                    TwitterData((X_test, y_test, id_test, g_test))

    if sensitive_attr == 'identity':
        num_sensitive_classes = num_ids
        twitter_collate_batch = twitter_collate_batch_identity
    elif sensitive_attr == 'gender':
        twitter_collate_batch = twitter_collate_batch_gender
        num_sensitive_classes = num_genders


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True,
                                   collate_fn=twitter_collate_batch)

    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False,
                                   collate_fn=twitter_collate_batch)
    return train_loader, test_loader, voc_size, num_ages, num_sensitive_classes


def load_yelp(batch_size=8):

    train_dataset, test_dataset = datasets.YelpReviewFull(root=ROOT_DATA_PATH)

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size,
                                collate_fn=yelp_collate_batch, shuffle=True, drop_last=True)

    test_loader = data.DataLoader(test_dataset, batch_size=batch_size,
                                collate_fn=yelp_collate_batch, shuffle=True, drop_last=True)

    return train_loader, test_loader

if __name__ == '__main__':
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    train_loader, test_loader, voc_size, num_sensitive_classes, num_ages = load_twitter(sensitive_attr='gender')
    print(voc_size, num_sensitive_classes, num_ages, len(train_loader))