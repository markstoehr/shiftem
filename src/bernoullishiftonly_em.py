#!/usr/bin/python

from __future__ import division
import argparse
import numpy as np
from template_speech_rec import configParserWrapper
import template_speech_rec.get_train_data as gtrd
import bernoullishiftonly_em_fast
import matplotlib.pyplot as plt
from scipy.stats import norm
from mpl_toolkits.axes_grid1 import ImageGrid

def cluster_underlying_data(Z,posteriors,start_time,template_length):
    """
    Parameters:
    ===========
    Z: 
        Underlying data that edge features were computed from
    posteriors:
        Posteriors over the cluster identity for each datum
    start_time:
        useful for knowing when the template starts in time
    template_length:
        how much of the data to use for the template
    """
    # check if the data has more dimensions than just points, length and a feature length
    if len(Z.shape) > 3:
        Z_feature_shape = Z.shape[2:]
        Z = Z.reshape(Z.shape[0],Z.shape[1],np.prod(Z_feature_shape))
        reshape=True
    else:
        reshape=False

    num_classes = posteriors.shape[1]
    feature_length = Z.shape[2]
    Z_templates = np.zeros((num_classes,template_length,feature_length),dtype=np.float64)
    bernoullishiftonly_em_fast.compute_template_float(Z,posteriors,Z_templates,start_time)

    if reshape:
        Z = Z.reshape( *(  
            (Z.shape[0],Z.shape[1])
            + Z_feature_shape))
    return Z_templates
    
    

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

def compute_template(X,posteriors,template_length,start_time):
    """
    """
    num_shifts = posteriors.shape[1]
    template = np.zeros(template_length)
    for template_entry in xrange(template_length):
        import pdb; pdb.set_trace()
        template[template_entry] = np.mean((X[:,start_time+template_entry:start_time+template_entry+num_shifts] * posteriors).sum(1),0)

    return template

def compute_bgd(X,posteriors,template_length,start_time):
    num_shifts=posteriors.shape[1]
    bgd_mean = 0
    num_data = np.float((X.shape[1] - template_length) * X.shape[0])
    for shift in xrange(num_shifts):
        front_sec = 0 if shift == 1 else np.dot(posteriors[:,shift],
                                                X[:,start_time:start_time+shift]).sum()
        back_sec = 0 if template_length+shift==X.shape[1] else np.dot(posteriors[:,shift],
                                                          X[:,start_time+template_length+shift:]).sum()
        bgd_mean += (front_sec + back_sec)/num_data

    return bgd_mean


def compute_likelihood_posteriors(X,class_shift_probs,
                                  posteriors,
                                  probs,
                                  log_template_odds,
                                  log_template_inv_prob,
                                  bgd_sums,
                                  start_time,
                                  min_prob,
                                  class_shift_type='unconstrained',
                                  min_shift_sigma=.8):
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
    posteriors[:] = 0



    bernoullishiftonly_em_fast.compute_template_loglikelihood(
        X,
        posteriors,
        bgd_sums,
        log_template_odds,
        log_template_inv_prob_constants,
        class_shift_log_probs,
        start_time)



    # for shift_id in xrange(num_shifts):
        
        # use_time= shift_id + start_time
        # front_bgd = np.zeros(num_data) if use_time==0 else bgd_sums[:,use_time-1]  
        # try:
        #     back_bgd = np.zeros(num_data) if use_time+template_length == data_length else bgd_sums[:,-1] - bgd_sums[:,use_time+template_length-1]
        # except: import pdb; pdb.set_trace()

        # print shift_id, np.mean(front_bgd+back_bgd)
        # print class_shift_log_probs[:,shift_id]

        # # compute the template likelihoods
        # posteriors[:,:,shift_id] = (
        #     np.dot(
        #         np.lib.stride_tricks.as_strided(
        #             X[:,use_time:use_time+template_length],
        #             strides=(X.strides[0],1),
        #             shape=(num_data,class_template_size)), 
        #         np.lib.stride_tricks.as_strided(
        #             log_template_odds,
        #             strides=(8*class_template_size,8),
        #             shape=(num_classes,class_template_size)).T) 
        #     + log_template_inv_prob_constants)

        

        # import pdb; pdb.set_trace()


        # # add in the background terms
        # posteriors[:,:,shift_id] += (
        #     np.lib.stride_tricks.as_strided(
        #         front_bgd,strides=(8,0),
        #         shape=(num_data,num_classes)) 
        #     + np.lib.stride_tricks.as_strided(
        #         back_bgd,strides=(8,0),
        #         shape=(num_data,num_classes)))
        
        # posteriors[:,:,shift_id] += class_shift_log_probs[:,shift_id]

    
        
    

    max_data_posteriors = posteriors.max(-1).max(-1)
    print "shift count vector"
    print (posteriors == np.lib.stride_tricks.as_strided(            max_data_posteriors,            strides=(8,0,0),            shape=(num_data,num_classes,num_shifts))).sum(0).sum(0)



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

    print class_shift_type
    if class_shift_type=='normal_independent':
        print 'normal_independent'
        # get the class sums by marginalizing over shifts
        class_probs = probs.sum(-1).sum(0)
        class_probs /= class_probs.sum()
        
        shift_counts = probs.sum(0).sum(0)
        idx = np.linspace(
            - (num_shifts-1)/2,(num_shifts-1)/2,num_shifts)
        shift_mean = np.dot(shift_counts, idx)/shift_counts.sum()


        shift_sigma_raw = np.sqrt(np.dot(shift_counts, (idx - shift_mean)**2)/shift_counts.sum())

        shift_sigma = max(shift_sigma_raw,        min_shift_sigma)
        print shift_mean, shift_sigma, min_shift_sigma, shift_sigma_raw

        shift_probs = norm.pdf(idx,scale=shift_sigma)
        class_shift_probs[:] = np.outer(class_probs,shift_probs)

    elif class_shift_type=='normal_dependent':
        print 'normal_dependent'
        # we estimate a mean and variance for each class
        # separately
        # get the class sums by marginalizing over shifts
        idx = np.linspace(
                - (num_shifts-1)/2,(num_shifts-1)/2,num_shifts)
        
        class_probs = probs.sum(-1).sum(0)
        class_probs /= class_probs.sum()
        
        for class_id in xrange(num_classes):
            shift_counts = probs[:,class_id].sum(0)
            shift_counts /= shift_counts.sum()
            shift_mean = np.dot(shift_counts, idx)
            shift_sigma = max(
                np.sqrt(np.dot(shift_counts, (idx - shift_mean)**2)
                    ),
                min_shift_sigma)
            print shift_mean, shift_sigma
            shift_probs = norm.pdf(idx,scale=shift_sigma)
            class_shift_probs[class_id,:] = class_probs[class_id]*shift_probs
        
    else:
        # class_shift_type=='unconstrained'
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
    
    print "posterior class counts"
    print posteriors.sum(-1).sum(0)
    
    return likelihoods
            



def main(args):
    """
    EM only with shift
    templates are generated based on the configuration script
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    if config_d['EMTRAINING']['random_seed'] is not None:
        np.random.seed(config_d['EMTRAINING']['random_seed'])
    
    num_shifts=config_d['EMTRAINING']['num_shifts']
    template_length=config_d['EMTRAINING']['template_length']
    start_time=config_d['EMTRAINING']['start_time']
    num_classes=config_d['EMTRAINING']['num_classes']

    min_prob=config_d['EMTRAINING']['min_prob']
    class_shift_min_prob = config_d['EMTRAINING']['class_shift_min_prob']
    X = np.load(args.i).astype(np.uint8)
    num_data = X.shape[0]
    X_shape = X.shape[2:]
    if len(X_shape) > 1:
        X = X.reshape(X.shape[0],X.shape[1],
                      np.prod(X_shape))
    # just a random initialization
    init_class_ids = np.random.randint(num_classes,size=num_data)


    
    posteriors = np.zeros((X.shape[0],num_classes,
                          num_shifts)) 
    probs = posteriors.copy()

    if config_d['EMTRAINING'].has_key('initialization') and config_d['EMTRAINING']['initialization']=='random_class_uniform_shift':
        for i, class_id in enumerate(init_class_ids):
            posteriors[i,class_id,:] = 1./(num_shifts)
    else:
        print 'shift spike at %d' % ((num_shifts-1)/2)
        for i, class_id in enumerate(init_class_ids):
            posteriors[i,class_id,(num_shifts-1)/2] = 1.


    

    # initialize the class and shift probs

    class_shift_probs = 1./(num_classes * num_shifts) * np.ones((num_classes,num_shifts))
    class_shift_mask = np.ones((num_classes,
                                num_shifts),dtype=bool)


    template = np.zeros((num_classes,
                         template_length,
                         X.shape[2]))


    if config_d['EMTRAINING']['initialization']=='template':
        template[:] = np.load(args.init_templates)
    else:
        bernoullishiftonly_em_fast.compute_template(X,posteriors,template,start_time)
        template = np.clip(template,min_prob,1-min_prob)
    

    background = np.zeros(X.shape[2])


    
    bernoullishiftonly_em_fast.compute_background(X,
                                                    posteriors,
                                                    background,
                                                    start_time,
                                                    template_length)


    background = np.clip(background,min_prob,1-min_prob)
    log_template_inv_prob = np.log(1-template)
    log_template_odds = np.log(template) - log_template_inv_prob
                                    

    
    bgd_sums = prep_bgd_sums(X,background)





    likelihood = np.sum(compute_likelihood_posteriors(X,class_shift_probs,
                                               posteriors,
                                               probs,
                                               log_template_odds,
                                               log_template_inv_prob,
                                               bgd_sums,
                                                      start_time, class_shift_min_prob,
                                                      class_shift_type=config_d['EMTRAINING']['class_shift_type'],
                                                      min_shift_sigma=config_d['EMTRAINING']['min_shift_sigma']))


    




    not_converged = True
    iteration = 0
    while not_converged:
        prev_template0 = template[0]
        print "iter= %d\tlikelihood = %g" % (iteration,likelihood)
        iteration += 1
        template[:] = 0
        bernoullishiftonly_em_fast.compute_template(X,posteriors,template,start_time)
        #            import pdb; pdb.set_trace()
        background[:] = 0
        bernoullishiftonly_em_fast.compute_background(X,
                                                    posteriors,
                                                    background,
                                                    start_time,
                                                    template_length)


        template = np.clip(template,min_prob,1-min_prob)
        background = np.clip(background,min_prob,1-min_prob)
        
        # plt.close('all');
        # for i in xrange(num_classes):
        #     plt.subplot(num_classes,1,i+1)
        #     plt.imshow(template[i].T,origin='lower',cmap='bone',
        #                interpolation='nearest')

        # plt.show()


        log_template_inv_prob = np.log(1-template)
        log_template_odds = np.log(template) - log_template_inv_prob
    
        bgd_sums = prep_bgd_sums(X,background)

        
        new_likelihood = np.sum(compute_likelihood_posteriors(X,class_shift_probs,
                                                       posteriors,
                                                       probs,
                                                       log_template_odds,
                                                       log_template_inv_prob,
                                                       bgd_sums,start_time, class_shift_min_prob,
                                                      class_shift_type=config_d['EMTRAINING']['class_shift_type'],
                                                      min_shift_sigma=config_d['EMTRAINING']['min_shift_sigma']))

        print (likelihood-new_likelihood)/likelihood, config_d['EMTRAINING']['tolerance']
        not_converged = ((likelihood-new_likelihood)/likelihood  > config_d['EMTRAINING']['tolerance'] or iteration < 10) and iteration < 1000

        likelihood = new_likelihood



    if args.out_templates is None:
        np.save('%stemplates.npy' % args.o, template)
    else:
        np.save(args.out_templates, template)
    if args.out_backgrounds is None:
        np.save('%sbackground.npy' % args.o, background)
    else:
        np.save(args.out_backgrounds, background)
    if args.out_posteriors is None:
        np.save('%sposteriors.npy' % args.o,posteriors)
    else:
        np.save(args.out_posteriors, posteriors)
    if args.out_class_shift_probs is None:
        np.save('%sclass_shift_probs.npy' % args.o,class_shift_probs)
    else:
        np.save(args.out_class_shift_probs,
                class_shift_probs)
    
    if args.visualize_templates:
        for c_id, c_template in enumerate(template):
            cur_template = c_template.reshape(

                *(
                    (c_template.shape[0],)
                    + X_shape ))



            for i in xrange(cur_template.shape[-1]):
                plt.close('all')
                plt.imshow(cur_template[:,:,i].T,
                       cmap='hot',
                       interpolation='nearest',
                       origin='lower',vmin=0,vmax=1)
                plt.axis('off')
                plt.savefig('%stemplates_%d_%d.png' % (args.o,
                                                       c_id,i),
                        bbox_inches='tight')


if __name__=="__main__":
    parser = argparse.ArgumentParser("""
""")
    parser.add_argument("-o",type=str,default=None,help="output file to save the position posteriors to"
                        )
    parser.add_argument('--init_templates',type=str,default=None,
                    help='input template file to initialize the templates to')
    parser.add_argument('--out_templates',type=str,default=None,
                        help='output file to save the templates to')
    parser.add_argument('--out_backgrounds',type=str,default=None,
                        help='output file to save the backgrounds to')
    parser.add_argument('--out_posteriors',type=str,default=None,
                        help='output file to save the posteriors to')
    parser.add_argument('--out_class_shift_probs',type=str,default=None,
                        help='output file to save the class_shift_probs to')
    parser.add_argument("-c",type=str,default='main.config',help="config file: default is config")
    parser.add_argument('-i',type=str,default='data/generated_X.npy',
                        help='data used for the EM')
    parser.add_argument('--visualize_templates',action='store_true',
                        help='whether to output visualizations of the templates')
    main(parser.parse_args())
