import warnings
import os
from cifar10_c_unlabeled_utils  import (
    precompute_clip_stl10_train_image_embeddings,
    precompute_clip_stl10_unlabeled_image_embeddings,
    precompute_clip_stl10_test_image_embeddings,
    precompute_clip_stl10_text_embeddings,
    train_resnet18_zero_shot,
    train_resnet18_linear_probe
)

def f_name(f):
    return f.__name__
def make_dir(root, target):
    if not os.path.exists(root + target):
        os.mkdir(root + target)
warnings.filterwarnings('ignore')
root = '/home/user1/ariel/federated_learning/kd_jetson_cifar_ood/'
make_dir(root, 'data/')
make_dir(root, 'data/cifar10_c_unlabeled/')
make_dir(root,'data/experiments/')
make_dir(root,'data/experiments/cifar10_c_unlabeled')


precompute_clip_stl10_train_image_embeddings()
print('\nFinished precompute_clip_stl10_train_image_embeddings())')
precompute_clip_stl10_unlabeled_image_embeddings()
print('\nFinished precompute_clip_stl10_unlabeled_image_embeddings()')
precompute_clip_stl10_test_image_embeddings()
print('\nFinished precompute_clip_stl10_test_image_embeddings()')
precompute_clip_stl10_text_embeddings()
print('\nFinished precompute_clip_stl10_text_embeddings()')
train_resnet18_linear_probe(f_name(train_resnet18_linear_probe))
print('\nFinished train_resnet18_linear_probe(f_name(train_resnet18_linear_probe))')
train_resnet18_zero_shot(f_name(train_resnet18_zero_shot))
print('\nFinished train_resnet18_zero_shot(f_name(train_resnet18_zero_shot))')
