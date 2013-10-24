#!/usr/bin/python

from __future__ import division
import argparse
import numpy as np
from template_speech_rec import configParserWrapper
import template_speech_rec.get_train_data as gtrd

def prep_bgd_sums(X,bgd_prob):
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
    num_data, data_length = X.shape
    bgd_sums = np.zeros((num_data,
                         data_length))
    log_bgd_inv_prob = np.log(1-bgd_prob)
    log_bgd_odds = np.log(bgd_prob) -log_bgd_inv_prob
    for datum_id, x in enumerate(X):
        likes = x*log_bgd_odds + log_bgd_inv_prob
        bgd_sums[datum_id][:] = np.cumsum(likes)
    
    return bgd_sums

def compute_likelihood_posteriors(X,shift_probs,
                                  posteriors,
                                  log_template_odds,
                                  log_template_inv_prob,
                                  bgd_sums):
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
    num_data, num_shifts = posteriors.shape
    assert num_data == X.shape[0]
    data_length = X.shape[1]
    template_length = len(log_template_odds)
    shift_log_probs = np.log(shift_probs)

    likelihood = 0


    for data_id, x in enumerate(X):
        
        for shift_id in xrange(len(posteriors[data_id])):
            front_bgd = 0 if shift_id==0 else bgd_sums[data_id,shift_id-1]
            # back goes from end of template to end of the data
            back_bgd = bgd_sums[data_id,-1] - bgd_sums[data_id,shift_id+template_length]
            posteriors[data_id,shift_id] = front_bgd + back_bgd + np.sum(x[shift_id:shift_id+template_length]*log_template_odds+log_template_inv_prob) + shift_log_probs[shift_id]
            
        likelihood += posteriors[data_id].sum()
        # do normalization
        max_posterior = posteriors[data_id].max()
        probs = np.exp(posteriors[data_id]-max_posterior)
        probs /= probs.sum()
        shift_probs += probs
        posteriors[data_id][:] = probs

    shift_probs /= shift_probs.sum()

    return likelihood/X.shape[0]
            

def main(args):
    """
    EM only with shift
    templates are generated based on the configuration script
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    

    X = np.load(args.i)
    posteriors = np.ones((X.shape[0],
                          config_d['TRAINDATA']['num_shifts'])) * 1./config_d['TRAINDATA']['num_shifts']

    # initialize the shift probs
    shift_probs = 1./config_d['TRAINDATA']['num_shifts'] * np.ones(config_d['TRAINDATA']['num_shifts'])

    bgd_prob = config_d['TRAINDATA']['background_means']
    template = np.ones(config_d['TRAINDATA']['template_length']) * config_d['TRAINDATA']['template_means']
    log_template_inv_prob = np.log(1-template)
    log_template_odds = np.log(template) - log_template_inv_prob
                                    

    
    bgd_sums = prep_bgd_sums(X,bgd_prob)

    likelihood = compute_likelihood_posteriors(X,shift_probs,
                                                       posteriors,
                                                       log_template_odds,
                                                       log_template_inv_prob,
                                               bgd_sums)



    not_converged = True
    while not_converged:
        print "likelihood = %g" % likelihood
        new_likelihood = compute_likelihood_posteriors(X,shift_probs,
                                                       posteriors,
                                                       log_template_odds,
                                                       log_template_inv_prob,
                                               bgd_sums)
        not_converged = np.abs(new_likelihood -likelihood)/np.abs(likelihood) < config_d['EMTRAINING']['tolerance']
        likelihood = new_likelihood

    import pdb; pdb.set_trace()    
    
    np.save('%s_posteriors.npy' % args.o,posteriors)
    np.save('%s_shift_probs.npy' % args.o,shift_probs)


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
""")
    parser.add_argument("-o",type=str,default=None,help="output file to save the position posteriors to"
                        )
    parser.add_argument("-c",type=str,default='main.config',help="config file: default is config")
    parser.add_argument('-i',type=str,default='data/generated_X.npy',
                        help='data used for the EM')
    main(parser.parse_args())
