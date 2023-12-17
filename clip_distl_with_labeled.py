from cifar10_c_utils import (precompute_clip_stl10_train_image_embeddings,
                         precompute_clip_stl10_test_image_embeddings,
                         precompute_clip_stl10_text_embeddings, train_resnet18_from_scratch,
                         train_resnet18_zero_shot_train_only, train_resnet18_linear_probe_train_only)

def f_name(f):
    return f.__name__

precompute_clip_stl10_train_image_embeddings()
precompute_clip_stl10_test_image_embeddings()
precompute_clip_stl10_text_embeddings()
train_resnet18_from_scratch()
train_resnet18_zero_shot_train_only(f_name(train_resnet18_zero_shot_train_only))
train_resnet18_linear_probe_train_only(f_name(train_resnet18_linear_probe_train_only))
