from typing import List
from datasets.bases import TorchvisionDataset
from datasets.mvtec import ADMvTec


DS_CHOICES = ('mnist', 'cifar10', 'fmnist', 'mvtec', 'imagenet', 'pascalvoc', 'custom', 'BTAD')
PREPROC_CHOICES = (
    'lcn', 'lcnaug1', 'aug1', 'aug1_blackcenter', 'aug1_blackcenter_inverted', 'none'
)
CUSTOM_CLASSES = []


def load_dataset(dataset_name: str, data_path: str, normal_class: int, preproc: str,
                 supervise_mode: str, noise_mode: str, online_supervision: bool, nominal_label: int, only_test=False
                 ) -> TorchvisionDataset:
    """ Loads the dataset with given preprocessing pipeline and supervise parameters """

    assert dataset_name in DS_CHOICES
    assert preproc in PREPROC_CHOICES

    dataset = None
    if dataset_name == 'mvtec' or 'BTAD':
        dataset = ADMvTec(
            root=data_path, normal_class=normal_class, preproc=preproc,
            supervise_mode=supervise_mode, noise_mode=noise_mode, online_supervision=online_supervision,
            nominal_label=nominal_label, only_test=only_test
        )
    else:
        raise NotImplementedError(f'Dataset {dataset_name} is unknown.')

    return dataset    # 返回的是类


def no_classes(dataset_name: str) -> int:   # 返回键值
    return {
        'cifar10': 10,
        'fmnist': 10,
        'mvtec': 15,
        'imagenet': 30,
        'pascalvoc': 1,
        'custom': len(CUSTOM_CLASSES)
    }[dataset_name]


def str_labels(dataset_name: str) -> List[str]:
    return {
        'mvtec': [
            'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
            'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
            'wood', 'zipper'
        ],

    }[dataset_name]
