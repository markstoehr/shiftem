#!/usr/bin/python

from __future__ import division
import argparse
import numpy as np
from template_speech_rec import configParserWrapper
import template_speech_rec.get_train_data as gtrd
import bernoullishiftonly_em_fast
import matplotlib.pyplot as plt

def prep_bgd_sums(X,background):
    """
    Purpose is to create an array
    `bgd_sums` with the property
    that `bgd_sums[i,t]` is equal to the
    likelihood of observing
    `X[i,:t+1]` under the background 
    model (an independent product of bernoullis
    model with mean `bgd_prob`)
    

    Parameters:
    ===========
    X: np.ndarray[ndim=2]
       Data
    bgd_prob: float
       background probability
    

    Returns:
    ========
    bgd_sums: np.ndarray[ndim=2]
    """
    num_data, data_length, num_features = X.shape
    bgd_sums = np.zeros((num_data,
                         data_length))
    log_bgd_inv_prob = np.log(1-background)
    log_bgd_odds = np.log(background) -log_bgd_inv_prob
    for datum_id, x in enumerate(X):
        likes = x*log_bgd_odds + log_bgd_inv_prob

        bgd_sums[datum_id][:] = np.cumsum(likes.sum(1))
    
    return bgd_sums


def compute_likelihood_posteriors(X,class_shift_probs,
                                  posteriors,
                                  probs,
                                  log_template_odds,
                                  log_template_inv_prob,
                                  bgd_sums,
                                  start_time,
                                  min_prob):
    """
    Parameters:
    ===========
    X: np.ndarray[ndim=2]
        Data with the slower index is over data points
        and the faster index is over entries of a datum
    posteriors: np.ndarray[ndim=1]
        index 0 ranges over different posterior weights over
        the start time for the object within the recording.
    template: np.ndarray[ndim=1]
    bgd_prob: float
    """
    num_data, num_classes, num_shifts = posteriors.shape
    assert num_data == X.shape[0]
    data_length = X.shape[1]
    template_length = log_template_odds.shape[1]
    num_features = log_template_odds.shape[2]
    class_template_size = template_length*num_features
    log_template_inv_prob_constants = log_template_inv_prob.sum(-1).sum(-1)
    class_shift_log_probs = np.log(class_shift_probs)
    probs[:] = 0




    for shift_id in xrange(num_shifts):

        use_time= shift_id + start_time
        front_bgd = np.zeros(num_data) if use_time==0 else bgd_sums[:,use_time-1]  
        try:
            back_bgd = np.zeros(num_data) if use_time+template_length == data_length else bgd_sums[:,-1] - bgd_sums[:,use_time+template_length-1]
        except: import pdb; pdb.set_trace()


        

        # compute the template likelihoods
        posteriors[:,:,shift_id] = (
            np.dot(
                np.lib.stride_tricks.as_strided(
                    X[:,use_time:use_time+template_length],
                    strides=(X.strides[0],1),
                    shape=(num_data,class_template_size)), 
                np.lib.stride_tricks.as_strided(
                    log_template_odds,
                    strides=(8*class_template_size,8),
                    shape=(num_classes,class_template_size)).T) 
            + log_template_inv_prob_constants)





        # add in the background terms
        posteriors[:,:,shift_id] += (
            np.lib.stride_tricks.as_strided(
                front_bgd,strides=(8,0),
                shape=(num_data,num_classes)) 
            + np.lib.stride_tricks.as_strided(
                back_bgd,strides=(8,0),
                shape=(num_data,num_classes)))
        
        posteriors[:,:,shift_id] += class_shift_log_probs[:,shift_id]

    
        


    max_data_posteriors = posteriors.max(-1).max(-1)
    probs = np.exp( 
        posteriors
        - np.lib.stride_tricks.as_strided(
            max_data_posteriors,
            strides=(8,0,0),
            shape=(num_data,num_classes,num_shifts)))
    
    probs_sum = probs.sum(-1).sum(-1)
    likelihoods = np.log(probs_sum) + max_data_posteriors
             
    # for x_id, x in enumerate(posteriors):
    #     max_class_shift = x.argmax()
    #     max_class = max_class_shift / num_shifts
    #     max_shift = max_class_shift % num_shifts

    #     probs[x_id][:] = 0.
    #     probs[x_id,max_class,max_shift] = 1.

    # print probs.sum(0)

    probs = np.exp(
        posteriors
        - np.lib.stride_tricks.as_strided(
            likelihoods,
            strides=(8,0,0),
            shape=(num_data,num_classes,num_shifts)))
    

    # class shift posteriors
    max_class_shift_posteriors = posteriors.max(0)

    class_shift_probs[:] = np.exp(np.clip(np.log(np.exp(posteriors - max_class_shift_posteriors).sum(0)) + max_class_shift_posteriors,
                                          np.log(min_prob),
                                          np.log(1-min_prob)))
    # max_class_shift_log_prob = class_shift_log_probs.max()
    
    # class_shift_probs[:] = 1./(num_classes * num_shifts)
    class_shift_sum = class_shift_probs.sum()
    # likelihood_class_shift_sum = np.sum(np.log(class_shift_sum) + max_class_shift_posteriors)
    class_shift_probs /= class_shift_sum


    # print class_shift_probs, likelihood_class_shift_sum
    posteriors[:] = probs
    return likelihoods
            


def compute_likelihood(data,
                       log_template_odds,
                       log_template_inv_probs,
                       bgd_sums,
                       start_time,
                       num_shifts):
    """
    """
    num_classes = log_template_odds.shape[0]
    num_data = data.shape[0]
    data_length = data.shape[1]
    template_length = log_template_odds.shape[1]
    num_features = log_template_odds.shape[2]
    class_template_size = template_length*num_features
    log_template_inv_prob_constants = log_template_inv_probs.sum(-1).sum(-1)

    likelihoods = np.zeros((num_data,
                           num_classes,
                           num_shifts))

    bernoullishiftonly_em_fast.compute_template_loglikelihood(
        data,
        likelihoods,
        bgd_sums,
        log_template_odds,
        log_template_inv_prob_constants,
        np.zeros((num_classes,num_shifts)),
        start_time)


    # for shift_id in xrange(num_shifts):

    #     use_time= shift_id + start_time
    #     front_bgd = np.zeros(num_data) if use_time==0 else bgd_sums[:,use_time-1]  
    #     try:
    #         back_bgd = np.zeros(num_data) if use_time+template_length == data_length else bgd_sums[:,-1] - bgd_sums[:,use_time+template_length-1]
    #     except: import pdb; pdb.set_trace()

    #     # compute the template likelihoods

    #     likelihoods[:,:,shift_id] = (
    #         np.dot(
    #             np.lib.stride_tricks.as_strided(
    #                 data[:,use_time:use_time+template_length],
    #                 strides=(data.strides[0],1),
    #                 shape=(num_data,class_template_size)), 
    #             np.lib.stride_tricks.as_strided(
    #                 log_template_odds,
    #                 strides=(8*class_template_size,8),
    #                 shape=(num_classes,class_template_size)).T) 
    #         + log_template_inv_prob_constants)





    #     # add in the background terms
    #     likelihoods[:,:,shift_id] += (
    #         np.lib.stride_tricks.as_strided(
    #             front_bgd,strides=(8,0),
    #             shape=(num_data,num_classes)) 
    #         + np.lib.stride_tricks.as_strided(
    #             back_bgd,strides=(8,0),
    #             shape=(num_data,num_classes)))
        

    
    
    
    return likelihoods.max(-1).max(-1)

def main(args):
    """
    EM only with shift
    templates are generated based on the configuration script
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    
    num_shifts=config_d['EMTRAINING']['num_shifts']
    num_models = len(args.models)
    assert num_models == len(args.data)

    models = []
    backgrounds = []

    template_log_inv_probs = []
    template_log_odds = []
    for i in xrange(num_models):
        models.append(np.load(args.models[i]))
        backgrounds.append(np.load(args.bgds[i]))

        template_log_inv_probs.append(np.log(1-models[i]))

        template_log_odds.append(np.log(models[i]) - template_log_inv_probs[i])


    data = []
    out_likelihoods = []
    for i in xrange(num_models):
        data.append(np.load(args.data[i]))
        likelihoods = []
        for j in xrange(num_models):
            bgd_sums = prep_bgd_sums(data[i],backgrounds[j])

            likelihoods.append(
                compute_likelihood(
                    data[i],
                    template_log_odds[j],
                    template_log_inv_probs[j],
                    bgd_sums,
                    config_d['INFERENCE']['start_time'],
                    config_d['INFERENCE']['num_shifts']))

        out_likelihoods.append(np.array(likelihoods))
        

    confusion_matrix = np.zeros((num_models,num_models))
    for data_id in xrange(num_models):
        for model_id in xrange(num_models):
            confusion_matrix[data_id,
                             model_id] = np.sum(out_likelihoods[data_id].argmax(0) == model_id)

    num_data = confusion_matrix.sum()
    error_rate = (num_data - np.diag(confusion_matrix).sum())/num_data
    print "error_rate = %f" % error_rate

    np.save('%sconfustion_matrix.npy' % args.out,confusion_matrix)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
""")
    parser.add_argument("--out",type=str,default=None,help="output file to save the classification results to"
                        )
    parser.add_argument("-c",type=str,default='main.config',help="config file: default is config")
    parser.add_argument('--models',type=str,nargs='+', help='sequence of models to be used should correspond to the sequence of files passed to the --data flag')
    parser.add_argument('--data',type=str,nargs='+',
                        help='sequence of data sources, assumed to be the data examples associated with a particular model instance')
    parser.add_argument('--bgds',type=str,nargs='+',
                        help='backgrounds')
    main(parser.parse_args())
