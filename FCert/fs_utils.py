import torch
import numpy as np
from typing import Tuple
import torch.nn.functional as F
import time

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
def cosine_dist(x,y):
    A=x
    B = y
    # Calculate the cosine similarity for each pair of vectors
    similarity = F.cosine_similarity(A.unsqueeze(1), B, dim=-1)
    # Calculate the cosine distance for each pair of vectors
    distance = 1 - similarity
    return distance

def cosine_similarity(x,y):
    A=x
    B = y
    # Calculate the cosine similarity for each pair of vectors
    similarity = F.cosine_similarity(A.unsqueeze(1), B, dim=-1)
    return similarity
def l2_dist(x,y):
    return torch.cdist(x, y, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary')
def l1_dist(x,y):
    return torch.cdist(x, y, p=1.0, compute_mode='use_mm_for_euclid_dist_if_necessary')

def get_distances(opt, input, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py

    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))
    support_samples = torch.flatten(torch.stack([input_cpu[idx_list] for idx_list in support_idxs]), start_dim=0, end_dim=1)
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    #input_cpu[idx_list]
    if opt.metric_type == "euclidean":
        dists = euclidean_dist(query_samples, support_samples)
    elif opt.metric_type =="cosine":
        dists = cosine_dist(query_samples, support_samples)
    elif opt.metric_type =="l1":
        dists = l1_dist(query_samples, support_samples)
    elif opt.metric_type =="l2":
        dists = l2_dist(query_samples, support_samples)
    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    return dists, target_inds


def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def get_feature(opt,model,x):
    with torch.no_grad():
        if opt.model_type == "CLIP":
            model_output = model.encode_image(x).float()
        else:
            print("model type not implemented")
    return model_output