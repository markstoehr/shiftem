# cython bernoullishiftonly_em_fast.pyx
# gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
#      -I/usr/include/python2.7 -o bernoullishiftonly_em_fast.so bernoullishiftonly_em_fast.c
#!python
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
import cython
import numpy as np
cimport numpy as np
#from cython.parallel import prange
DTYPE = np.float64
UINT = np.uint8
ctypedef np.float64_t DTYPE_t

ctypedef np.uint8_t UINT_t

def compute_template_loglikelihood(np.ndarray[ndim=3,
                                              dtype=UINT_t] X,
                                   np.ndarray[ndim=3,dtype=DTYPE_t] posteriors,
                                   np.ndarray[ndim=2,dtype=DTYPE_t] bgd_sums,
                                   np.ndarray[ndim=3,dtype=DTYPE_t] template_log_odds,
                                   np.ndarray[ndim=1,dtype=DTYPE_t] template_log_inv_prob_sums,
                                   np.ndarray[ndim=2,dtype=DTYPE_t] class_shift_log_probs,
                                   unsigned int start_time):
    """
    X:
      data
    posteriors:
       where raw likelihoods go
    bgd_sums:
       integral transform of the background for fast computation
    template_log_odds:
       templates
    """
    cdef unsigned int num_data = X.shape[0]
    cdef unsigned int num_times = X.shape[1]
    cdef unsigned int num_features = X.shape[2]
    cdef unsigned int num_classes = posteriors.shape[1]
    cdef unsigned int num_shifts = posteriors.shape[2]
    cdef unsigned int template_length = template_log_odds.shape[1]

    cdef unsigned int x_id,c_id,t_id,s_id,f_id, use_time

    for x_id in range(num_data):
        for c_id in range(num_classes):
            for s_id in range(num_shifts):
                posteriors[x_id,c_id,s_id] += template_log_inv_prob_sums[c_id]
                use_time = s_id + start_time
                end_time = use_time + template_length
                
                # handle backgrounds
                # front background
                if use_time > 0:
                    posteriors[x_id,c_id,s_id] += bgd_sums[x_id,
                                                           use_time-1]
                # back background
                if end_time < num_times:
                    posteriors[x_id,c_id,s_id] += bgd_sums[x_id,
                                                           num_times-1] - bgd_sums[x_id,end_time-1]
                    
                for t_id in range(template_length):
                    for f_id in range(num_features):
                        if X[x_id,use_time+t_id,f_id] > 0:
                            posteriors[x_id,c_id,s_id] += template_log_odds[c_id,t_id,f_id]

                posteriors[x_id,c_id,s_id] += class_shift_log_probs[c_id,s_id]
                        


def compute_template_loglikelihood(np.ndarray[ndim=3,
                                              dtype=UINT_t] X,
                                   np.ndarray[ndim=3,dtype=DTYPE_t] posteriors,
                                   np.ndarray[ndim=2,dtype=DTYPE_t] bgd_sums,
                                   np.ndarray[ndim=3,dtype=DTYPE_t] template_log_odds,
                                   np.ndarray[ndim=1,dtype=DTYPE_t] template_log_inv_prob_sums,
                                   np.ndarray[ndim=2,dtype=DTYPE_t] class_shift_log_probs,
                                   unsigned int start_time):
    """
    X:
      data
    posteriors:
       where raw likelihoods go
    bgd_sums:
       integral transform of the background for fast computation
    template_log_odds:
       templates
    """
    cdef unsigned int num_data = X.shape[0]
    cdef unsigned int num_times = X.shape[1]
    cdef unsigned int num_features = X.shape[2]
    cdef unsigned int num_classes = posteriors.shape[1]
    cdef unsigned int num_shifts = posteriors.shape[2]
    cdef unsigned int template_length = template_log_odds.shape[1]

    cdef unsigned int x_id,c_id,t_id,s_id,f_id, use_time

    for x_id in range(num_data):
        for c_id in range(num_classes):
            for s_id in range(num_shifts):
                posteriors[x_id,c_id,s_id] += template_log_inv_prob_sums[c_id]
                use_time = s_id + start_time
                end_time = use_time + template_length
                
                # handle backgrounds
                # front background
                if use_time > 0:
                    posteriors[x_id,c_id,s_id] += bgd_sums[x_id,
                                                           use_time-1]
                # back background
                if end_time < num_times:
                    posteriors[x_id,c_id,s_id] += bgd_sums[x_id,
                                                           num_times-1] - bgd_sums[x_id,end_time-1]
                    
                for t_id in range(template_length):
                    for f_id in range(num_features):
                        if X[x_id,use_time+t_id,f_id] > 0:
                            posteriors[x_id,c_id,s_id] += template_log_odds[c_id,t_id,f_id]

                posteriors[x_id,c_id,s_id] += class_shift_log_probs[c_id,s_id]
                        
                                   
                                              

def compute_background(np.ndarray[ndim=3,dtype=UINT_t] X,
                       np.ndarray[ndim=3,dtype=DTYPE_t] posteriors,
                       np.ndarray[ndim=1,dtype=DTYPE_t] background,
                       unsigned int start_time,
                       unsigned int template_length):
    """
    X:
        Data, slow index are the different images, next index is time,
        and the fastest index is over features
    posteriors:
        Posterior distribution over shift, slow index is over data points
        fast index is over shifts

    background:
        index is over features--background is constant with time

    start_time: int
        when we should start X
    """ 
    cdef unsigned int num_data = X.shape[0]
    cdef unsigned int num_times = X.shape[1]
    cdef unsigned int num_features = X.shape[2]
    cdef unsigned int num_classes = posteriors.shape[1]
    cdef unsigned int num_shifts = posteriors.shape[2]

    cdef unsigned int x_id,c_id,t_id,s_id,f_id

    cdef unsigned int num_bgd_times = num_data * (num_times-template_length)


    for x_id in range(num_data):
        for c_id in range(num_classes):
            for s_id in range(num_shifts):
                # handle the front section of background times
                if s_id + start_time > 0:
                    for t_id in range(s_id+start_time):
                        for f_id in range(num_features):
                            background[f_id] += posteriors[x_id,c_id,s_id] * X[x_id,t_id,f_id]
                        
                # training background times
                if s_id + start_time + template_length < num_times:
                    for t_id in range(s_id+start_time+template_length,
                                  num_times):
                        for f_id in range(num_features):
                            background[f_id] += posteriors[x_id,c_id,s_id] * X[x_id,t_id,f_id]


    for f_id in range(num_features):
        background[f_id] /= num_bgd_times
                    

    

def compute_template(np.ndarray[ndim=3,dtype=UINT_t] X,
                     np.ndarray[ndim=3,dtype=DTYPE_t] posteriors,
                     np.ndarray[ndim=3,dtype=DTYPE_t] template,
                     unsigned int start_time):
    """
    X:
        Data, slow index are the different images, next index is time,
        and the fastest index is over features
    posteriors:
        Posterior distribution over shift, slow index is over data points
        fast index is over shifts

    template:
        template, slow index is over time and should range over
        template length, fast index is over features
    start_time: int
        when we should start X
    """ 

    cdef unsigned int num_data = X.shape[0]
    cdef unsigned int num_features = X.shape[2]
    cdef unsigned int num_classes = posteriors.shape[1]
    cdef unsigned int num_shifts = posteriors.shape[2]
    cdef unsigned int template_length = template.shape[1]

    cdef unsigned int c_id,x_id,t_id,s_id,f_id
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] class_sums = np.zeros(num_classes,
                                                                dtype=DTYPE)


    for x_id in range(num_data):
        for c_id in range(num_classes):
            for s_id in range(num_shifts):
                for t_id in range(template_length):
                    for f_id in range(num_features):
                        if X[x_id,start_time+t_id+s_id,f_id] > 0:
                            template[c_id,t_id,f_id] +=  posteriors[x_id,c_id,s_id]

                class_sums[c_id] += posteriors[x_id,c_id,s_id]
    

    for c_id in range(num_classes):
        for t_id in range(template_length):
            for f_id in range(num_features):
                template[c_id,t_id,f_id] /= class_sums[c_id]

def compute_template_float(np.ndarray[ndim=3,dtype=DTYPE_t] X,
                     np.ndarray[ndim=3,dtype=DTYPE_t] posteriors,
                     np.ndarray[ndim=3,dtype=DTYPE_t] template,
                     unsigned int start_time):
    """
    X:
        Data, slow index are the different images, next index is time,
        and the fastest index is over features
    posteriors:
        Posterior distribution over shift, slow index is over data points
        fast index is over shifts

    template:
        template, slow index is over time and should range over
        template length, fast index is over features
    start_time: int
        when we should start X
    """ 

    cdef unsigned int num_data = X.shape[0]
    cdef unsigned int num_features = X.shape[2]
    cdef unsigned int num_classes = posteriors.shape[1]
    cdef unsigned int num_shifts = posteriors.shape[2]
    cdef unsigned int template_length = template.shape[1]

    cdef unsigned int c_id,x_id,t_id,s_id,f_id
    cdef np.ndarray[ndim=1,dtype=DTYPE_t] class_sums = np.zeros(num_classes,
                                                                dtype=DTYPE)


    for x_id in range(num_data):
        for c_id in range(num_classes):
            for s_id in range(num_shifts):
                for t_id in range(template_length):
                    for f_id in range(num_features):
                        template[c_id,t_id,f_id] +=  posteriors[x_id,c_id,s_id] * X[x_id,start_time+t_id+s_id,f_id]

                class_sums[c_id] += posteriors[x_id,c_id,s_id]
    

    for c_id in range(num_classes):
        for t_id in range(template_length):
            for f_id in range(num_features):
                template[c_id,t_id,f_id] /= class_sums[c_id]
