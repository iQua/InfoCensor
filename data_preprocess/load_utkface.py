import glob
import os
import torch
from datetime import datetime
from torch.utils import data
from torchvision.datasets.folder import pil_loader
from torchvision import transforms

ROOT_DATA_PATH = '/opt/data/zth/utkface/UTKFace'

class UTKFace(data.Dataset):

    gender_map = dict(male=0, female=1)
    race_map = dict(white=0, black=1, asian=2, indian=3, others=4)

    def __init__(self, root=ROOT_DATA_PATH, transform=None,
                        target_attr='gender', sensitive_attr='race'):
        self.root = root
        self.transform = transform
        self.samples = self._prepare_samples(root)

        self.target_attr = target_attr
        self.sensitive_attr = sensitive_attr

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label[self.target_attr], label[self.sensitive_attr]

    def __len__(self):
        return len(self.samples)

    def _prepare_samples(self, root):
        samples = []

        paths = glob.glob(os.path.join(root, '*.jpg'))

        for path in paths:
            try:
                label = self._load_label(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue
            if label['race'] != 4:
                samples.append((path, label))

        return samples

    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        age, gender, race = map(int, str_list[:3])
        label = dict(age=self._bin_age(age), gender=gender, race=race)
        return label

    def _load_datetime(self, s):
        return datetime.strptime(s, '%Y%m%d%H%M%S%f')

    def _bin_age(self, age):
        return min(age//10, 9)



class UTKFace_full(data.Dataset):

    gender_map = dict(male=0, female=1)
    race_map = dict(white=0, black=1, asian=2, indian=3, others=4)

    def __init__(self, root=ROOT_DATA_PATH, transform=None,
                        target_attr='age'):
        self.root = root
        self.transform = transform
        self.samples = self._prepare_samples(root)

        self.target_attr = target_attr

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = pil_loader(path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label['age'], label['gender'], label['race']

    def __len__(self):
        return len(self.samples)

    def _prepare_samples(self, root):
        samples = []

        paths = glob.glob(os.path.join(root, '*.jpg'))

        for path in paths:
            try:
                label = self._load_label(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue
            if label['race'] != 4:
                samples.append((path, label))

        return samples

    def _load_label(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        age, gender, race = map(int, str_list[:3])
        label = dict(age=self._bin_age(age), gender=gender, race=race)
        return label

    def _load_datetime(self, s):
        return datetime.strptime(s, '%Y%m%d%H%M%S%f')

    def _bin_age(self, age):
        return min(age//10, 9)

def load_utkface(img_dim=50, batch_size=128, random_seed=0, train_size=0.8,
                    target_attr='gender', sensitive_attr='race'):

    img_transform = transforms.Compose([
            transforms.Resize(img_dim),
            transforms.ToTensor(),
        ])

    dataset = UTKFace(root=ROOT_DATA_PATH, transform=img_transform,
                        target_attr=target_attr, sensitive_attr=sensitive_attr)
    train_dataset, test_dataset = data.random_split(dataset, [int(train_size*len(dataset)), len(dataset)-int(train_size*len(dataset))],
                                    generator=torch.Generator().manual_seed(random_seed))

    num_classes = dict(age=10, gender=2, race=4)


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False)

    return train_loader, test_loader, img_dim, num_classes[target_attr], num_classes[sensitive_attr]

def load_utkface_attack(img_dim=50, batch_size=128, random_seed=0, train_size=0.8, attacker_size=0.5,
                    target_attr='gender', sensitive_attr='race'):

    img_transform = transforms.Compose([
            transforms.Resize(img_dim),
            transforms.ToTensor(),
        ])

    dataset = UTKFace(root=ROOT_DATA_PATH, transform=img_transform,
                        target_attr=target_attr, sensitive_attr=sensitive_attr)
    train_dataset, test_dataset = data.random_split(dataset, [int(train_size*len(dataset)), len(dataset)-int(train_size*len(dataset))],
                                    generator=torch.Generator().manual_seed(random_seed))

    train_dataset, _ = data.random_split(train_dataset, [int(attacker_size*len(train_dataset)), len(train_dataset)-int(attacker_size*len(train_dataset))],
                                    generator=torch.Generator().manual_seed(random_seed))

    num_classes = dict(age=10, gender=2, race=4)


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False)

    return train_loader, test_loader, img_dim, num_classes[target_attr], num_classes[sensitive_attr]



def load_utkface_full(img_dim=50, batch_size=128, random_seed=0, train_size=0.8):

    img_transform = transforms.Compose([
            transforms.Resize(img_dim),
            transforms.ToTensor(),
        ])

    dataset = UTKFace_full(root=ROOT_DATA_PATH, transform=img_transform)
    train_dataset, test_dataset = data.random_split(dataset, [int(train_size*len(dataset)), len(dataset)-int(train_size*len(dataset))],
                                    generator=torch.Generator().manual_seed(random_seed))

    num_classes = dict(age=10, gender=2, race=4)


    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size, shuffle=True,
                                   num_workers=2,
                                   drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=2,
                                   drop_last=False)

    return train_loader, test_loader, img_dim, num_classes['age'], num_classes['gender'], num_classes['race']

if __name__ == '__main__':
    train_loader, test_loader, input_dim, target_num_classes, sensitive_num_classes = load_utkface(target_attr='age', sensitive_attr='gender')
