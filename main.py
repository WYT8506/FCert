import torch
#from prototypical_batch_sampler import PrototypicalBatchSampler
from data import *
from utils import *
from parser_util import get_parser
from torchvision import transforms
import numpy as np
import torch
import numpy as np
import clip
from FCert import *
import json

def init_dataset(opt, mode):
    print("mode:",mode)
    dataset_name = opt.dataset_type
    size = 224
    if dataset_name =="cifarfs":
        dataset = CIFARFS(mode=mode,root='./data',transform =  transforms.Compose([
                    transforms.Resize(size),
                    transforms.CenterCrop(size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([0.4914, 0.4822, 0.4465]),
                                                     np.array([0.2023, 0.1994, 0.2010]))
                  ]),download = True)
        
    if dataset_name =="cubirds200":
        dataset = CUBirds200(mode=mode,root='./data',transform =  transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
          ]),download = True)

    if dataset_name == "tiered_imagenet":
        dataset = TieredImagenet(mode=mode,root='./data',transform =  transforms.Compose([
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                             np.array([0.229, 0.224, 0.225]))
              ]),download = True)

    labels = []
    for i in range(len(dataset)):
        labels.append(dataset[i][1])
    n_classes = len(np.unique(labels))
    dataset.y = labels
    print("n_classes:", n_classes)
    
    if n_classes < opt.classes_per_it_val:
        raise(Exception('There are not enough classes in the dataset in order ' +
                        'to satisfy the chosen classes_per_it. Decrease the ' +
                        'classes_per_it_{tr/val} option and try again.'))
    return dataset


def init_sampler(opt, labels, mode):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val

        num_samples = opt.num_support_val + opt.num_query_val

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations)


def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)
    return dataloader


def main():
    '''
    Initialize everything
    '''
    options = get_parser().parse_args()

    K = options.num_support_val
    C = options.classes_per_it_val
    Kp = int((K-1)/2) #k'
    CERTIFICATION_TYPE = options.certification_type
    MODEL_TYPE =  options.model_type
    DATASET_TYPE = options.dataset_type

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)

    test_dataloader = init_dataloader(options, 'test')
    if MODEL_TYPE == "CLIP":
        model, preprocess = clip.load('ViT-B/32', 'cuda')
    else:
        print("Invalid model type")
    # Initialize the saved dictionary

    my_dict = {
    "K": K,
    "C": C,
    "Kp": Kp,
    "certification_type":CERTIFICATION_TYPE,
    "model_type":MODEL_TYPE,
    "dataset_type":DATASET_TYPE,
    "fcert": [0 for _ in range(Kp+2)]
    }

    with torch.no_grad():
        print(f"Attack type: {CERTIFICATION_TYPE}, Model: {MODEL_TYPE}, Dataset: {DATASET_TYPE}, K={K}, C={C}")
        print("FCert:")
        for t in range(Kp+2):
            acc = certify(opt=options,
                test_dataloader=test_dataloader,
                model=model,clip_k=Kp,T =t,certification_type = CERTIFICATION_TYPE)
            my_dict['fcert'][t]=acc
            print("Poisoning size:", t, " Certified Accuracy: ", acc)
        with open(options.file_path, "w") as json_file:
            json.dump(my_dict, json_file)
    
if __name__ == '__main__':
    main()