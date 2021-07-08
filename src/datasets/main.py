from .mnist import MNIST_Dataset
from .cifar10 import CIFAR10_Dataset
#  from .imdb import IMDB_Dataset
from .news20 import News20_Dataset
from .trec import Trec_Dataset


def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist', 'cifar10', 'news20', 'trec')
    assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    #  if dataset_name == 'imdb':
    #      dataset = IMDB_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'news20':
        dataset = News20_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'trec':
        dataset = Trec_Dataset(root=data_path, normal_class=normal_class)

    return dataset
