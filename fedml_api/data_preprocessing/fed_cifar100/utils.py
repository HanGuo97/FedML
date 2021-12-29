import torch
import torchvision.transforms as transforms
from typing import Callable, Tuple, Optional

'''
preprocess reference : https://github.com/google-research/federated/blob/master/utils/datasets/cifar100_dataset.py
'''

def cifar100_transform(img_mean, img_std, train = True, crop_size = (24,24)):
    """cropping, flipping, and normalizing."""
    if train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=img_mean, std=img_std),
        ])


def preprocess_cifar_img(img, train):
    # scale img to range [0,1] to fit ToTensor api
    img = torch.div(img, 255.0)
    transoformed_img = torch.stack([cifar100_transform
        (i.type(torch.DoubleTensor).mean(),
            i.type(torch.DoubleTensor).std(),
            train)
        (i.permute(2,0,1)) 
        for i in img])
    return transoformed_img


def get_cifar_preprocess_fn(distort: bool) -> Callable[[torch.Tensor], torch.Tensor]:
    # We can also apply no distortions to the training data.
    # as is done in some of Google's federated implementations.
    def preprocess_fn(X: torch.Tensor) -> torch.Tensor:
        # scale img to range [0,1] to fit ToTensor api
        X = torch.div(X, 255.0)
        fn = cifar100_transform(
            X.type(torch.DoubleTensor).mean(),
            X.type(torch.DoubleTensor).std(),
            train=distort)
        return fn(X.permute(2, 0, 1))

    return preprocess_fn


class CIFAR100Dataset(torch.utils.data.Dataset):

    def __init__(
            self,
            Xs: torch.Tensor,
            Ys: torch.Tensor,
            transform: Optional[Callable] = None,
    ) -> None:
        if Xs.shape[0] != Ys.shape[0]:
            raise ValueError("Xs and Ys should have the same length.")
        if not callable(transform):
            raise ValueError("transform should be callable.")

        self.Xs = Xs
        self.Ys = Ys
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        _X = self.Xs[index]
        _Y = self.Ys[index]

        if self.transform is not None:
            _X = self.transform(_X)

        return _X, _Y

    def __len__(self) -> int:
        return self.Xs.shape[0]
