#!/usr/bin/python

import argparse
import numpy as np
from template_speech_rec import configParserWrapper
import template_speech_rec.get_train_data as gtrd

def main(args):
    """
    EM only with shift
    templates are generated based on the configuration script
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    X = np.load(args.i)
    
    
    np.random.seed(config_d['TRAINDATA']['random_seed'])
    # uniformly sample the start times
    start_times = (np.random.rand(config_d['TRAINDATA']['num_training']) * config_d['TRAINDATA']['num_shifts']).astype(int)


    for i,x in enumerate(X):
        x[:] = (np.random.rand(config_d['TRAINDATA']['vector_length']) < config_d['TRAINDATA']['background_means']).astype(np.uint8)
        # 2 indicates that its a binary valued random variable
        x[start_times[i]:start_times[i]+config_d['TRAINDATA']['template_length']] = (np.random.rand(config_d['TRAINDATA']['template_length']) < config_d['TRAINDATA']['template_means']).astype(np.uint8)


    np.save('%s_X.npy' % args.o,X)
    np.save('%s_start_times.npy' % args.o,start_times)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
""")
    parser.add_argument("-o",type=str,default=None,help="output file to save the position posteriors to"
                        )
    parser.add_argument("-c",type=str,default='main.config',help="config file: default is config")
    parser.add_argument('-i',type=str,default='data/generated_X.npy',
                        help='data used for the EM')
    main(parser.parse_args())
