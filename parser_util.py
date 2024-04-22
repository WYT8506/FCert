# coding=utf-8
import os
import argparse

default_K= 10
default_C= 10

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', '--dataset_root',
                        type=str,
                        help='path to dataset',
                        default='..' + os.sep + 'dataset')

    parser.add_argument('-cTr', '--classes_per_it_tr',
                        type=int,
                        help='number of random classes per episode for training, default=60',
                        default=5)
    parser.add_argument('-its', '--iterations',
                        type=int,
                        help='number of episodes per epoch, default=100',
                        default=100)
    parser.add_argument('-nsTr', '--num_support_tr',
                        type=int,
                        help='number of samples per class to use as support for training, default=5',
                        default=5)

    parser.add_argument('-nqTr', '--num_query_tr',
                        type=int,
                        help='number of samples per class to use as query for training, default=5',
                        default=5)

    parser.add_argument('-cVa', '--classes_per_it_val',
                        type=int,
                        help='number of random classes per episode for validation, default=5',
                        default=default_C)

    parser.add_argument('-nsVa', '--num_support_val',
                        type=int,
                        help='number of samples per class to use as support for validation, default=5',
                        default=default_K)

    parser.add_argument('-nqVa', '--num_query_val',
                        type=int,
                        help='number of samples per class to use as query for validation, default=1',
                        default=1)
    

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        default = True,
                        help='enables cuda')
    parser.add_argument('-ct', '--certification_type',
                    type=str,
                    help='',
                    default="ind")

    parser.add_argument('-abt', '--ablation_type',
                type=str,
                help='',
                default='kp')
    parser.add_argument('-metric', '--metric_type',
            type=str,
            help='',
            default='euclidean')

    parser.add_argument('-mt', '--model_type',
            type=str,
            help='',
            default="CLIP")
    parser.add_argument('-dt', '--dataset_type',
            type=str,
            help='',
            default="cubirds200")
    parser.add_argument('-fp', '--file_path',
            type=str,
            help='',
            default='./output/output.txt')

    return parser