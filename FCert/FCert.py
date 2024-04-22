import torch
import numpy as np
from .fs_utils import *

def certify(opt, test_dataloader, model,clip_k=4,T =1,certification_type = "ind",num_epochs =1,num_batches =50):
    #certification fails if T>k'
    if T>clip_k:
        return 0

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    model = model.to(device)
    avg_acc = list()
    batch_count = 0

    for epoch in range(num_epochs):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            batch_count+=1
            if batch_count ==num_batches:
                break
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            model_output = get_feature(opt,model,x)
            dist, target_inds = get_distances(opt,model_output, target=y,
                             n_support=opt.num_support_val)
            dist = torch.reshape(dist, (dist.size(0), opt.classes_per_it_val,opt.num_support_val))
            sorted_dist, _ = torch.sort(dist,dim = -1)

            #certify for individual attack
            if certification_type == "ind":
                # Calculate distance upper and lower bounds for each class
                clipped_dist_upper = sorted_dist[:,:,clip_k+T:sorted_dist.shape[2]-clip_k+T]
                clipped_dist_lower = sorted_dist[:,:,clip_k-T:sorted_dist.shape[2]-clip_k-T]
                clipped_dist = sorted_dist[:,:,clip_k:sorted_dist.shape[2]-clip_k]
                # Compute the mean of the clipped tensor
                clipped_mean = torch.mean(clipped_dist,dim = -1)
                clipped_mean_upper = torch.mean(clipped_dist_upper,dim = -1)
                clipped_mean_lower = torch.mean(clipped_dist_lower,dim = -1)
                bounds = torch.zeros_like(clipped_mean)

                # Take the upper bound for the ground-truth class, take lower bounds for other classes
                for i in range(clipped_mean.shape[0]):
                    for j in range(clipped_mean.shape[1]):
                        if torch.flatten(target_inds)[i] == j:
                            bounds[i][j] = clipped_mean_upper[i][j]
                        else:
                            bounds[i][j] = clipped_mean_lower[i][j]

                preds_poisoned = torch.argmin(bounds, dim = -1)
                correct_count = 0
                for i in range(torch.flatten(target_inds).shape[0]):
                    if preds_poisoned[i] == torch.flatten(target_inds)[i]: 
                        correct_count+=1

                acc = correct_count/torch.flatten(target_inds).shape[0]

            #certify for group attack
            if certification_type == "group":
                clipped_dist = sorted_dist[:,:,clip_k:sorted_dist.shape[2]-clip_k]
                clipped_mean = torch.mean(clipped_dist,dim = -1)
                all_clipped_mean_upper = []
                all_clipped_mean_lower = []
                
                for t in range(T+1):
                    clipped_dist_upper = sorted_dist[:,:,clip_k+t:sorted_dist.shape[2]-clip_k+t]
                    clipped_dist_lower = sorted_dist[:,:,clip_k-t:sorted_dist.shape[2]-clip_k-t]
                    clipped_mean_upper = torch.mean(clipped_dist_upper,dim = -1)
                    clipped_mean_lower = torch.mean(clipped_dist_lower,dim = -1)
                    all_clipped_mean_upper.append(clipped_mean_upper)
                    all_clipped_mean_lower.append(clipped_mean_lower)
                all_preds = []

                #Suppose the attacker modifies t1 data samples in the ground-truth class, and t2 data samples in the non-ground-truth class
                #Consider all combinations of t1 and t2
                for t1 in range(T+1):
                    t2 = T -t1
                    bounds = torch.zeros_like(clipped_mean)
                    for i in range(clipped_mean.shape[0]):
                        for j in range(clipped_mean.shape[1]):
                            if torch.flatten(target_inds)[i] == j:
                                bounds[i][j] = all_clipped_mean_upper[t1][i][j]
                            else:
                                bounds[i][j] = all_clipped_mean_lower[t2][i][j]
                    preds = torch.argmin(bounds, dim = -1)
                    all_preds.append(preds)
           
                all_preds = torch.stack(all_preds, dim = 0)
                preds_poisoned = torch.zeros(all_preds.shape[1])
                all_preds = torch.transpose(all_preds, 0, 1)
                
                for i in range(all_preds.shape[0]):
                    pred = all_preds[i]
                    if torch.all(torch.eq(pred, pred[0])):
                        preds_poisoned[i] = pred[0]
                    else:
                        preds_poisoned[i] = -1
                correct_count = 0

                for i in range(torch.flatten(target_inds).shape[0]):
                    if preds_poisoned[i] == torch.flatten(target_inds)[i]: 
                        correct_count+=1

                acc = correct_count/torch.flatten(target_inds).shape[0]
            
            avg_acc.append(acc)
    avg_acc = np.mean(avg_acc)
    return avg_acc

