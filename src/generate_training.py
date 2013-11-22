#!/usr/bin/python

import argparse
import numpy as np
from template_speech_rec import configParserWrapper
import template_speech_rec.get_train_data as gtrd
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def quadratic_curves_template(template_length,num_features,template_means,background_means, template_sigma):
    """
    Generate three random points one on each side of the template and (uniformly along the feature axis) and one inside the template (chosen uniformly), fit a quadratic between them and blur a template along those points
    """
    points = np.zeros((2,3),dtype=int)
    # get the x coordinates
    points[0,2] = template_length-1
    points[0,1] = np.random.randint(1,template_length-1)
    # get the coordinate along the feature axis
    points[1] = np.random.randint(num_features,size=points.shape[1])


    #solve for the quadratic coefficients
    X = np.ones((points.shape[1],
                 points.shape[1]))
    X[1] = points[0]
    X[2] = points[0] * points[0]
    coeffs = np.linalg.solve(X.T,points[1])
    all_X = np.ones((3,template_length))
    all_X[1] = np.arange(template_length)
    all_X[2] = all_X[1]**2
    y_coords = np.clip((.5 + np.dot(all_X.T,coeffs)).astype(int),
                       0,
                       num_features-1).astype(int)
    
    template = background_means * np.ones((template_length,
                                           num_features))
    for x,y in enumerate(y_coords):
        template[x,y] = template_means

    # now we blur using a standard filter
    return gaussian_filter(template,template_sigma)
    

def main(args):
    """
    Generate the training vectors with a random start time
    base everything on the config file
    
    """
    config_d = configParserWrapper.load_settings(open(args.c,'r'))

    X = np.zeros((config_d['TRAINDATA']['num_training'],
                  config_d['TRAINDATA']['vector_length'],
                  config_d['TRAINDATA']['num_features']),
                 dtype=np.uint8)

    template_lengths = config_d['TRAINDATA']['template_lengths']
    single_template_length = type(template_lengths) == int
    max_template_length = np.max(template_lengths)
    
    
    if config_d['TRAINDATA']['random_seed'] is not None:
        np.random.seed(config_d['TRAINDATA']['random_seed'])


    # uniformly sample the start times
    start_times = (np.random.rand(config_d['TRAINDATA']['num_training']) * config_d['TRAINDATA']['num_shifts']).astype(int)

    true_templates = np.zeros(
        (config_d['TRAINDATA']['num_classes'],
         max_template_length,config_d['TRAINDATA']['num_features']))

    # generate true templates
    if config_d['TRAINDATA']['template_type'] == 'uniform':
        true_templates += 1
        true_templates *= config_d['TRAINDATA']['background_means']
        feature_block_size = config_d['TRAINDATA']['num_features']/config_d['TRAINDATA']['num_classes']
        for class_id in xrange(config_d['TRAINDATA']['num_classes']):
            if single_template_length:
                cur_template_length = template_lengths
            else:
                cur_template_length = template_lengths[class_id]
            feature_block_start = class_id*feature_block_size
            feature_block_end = min((class_id+1)*feature_block_size,
                                    config_d['TRAINDATA']['num_features'])
            num_block_features = feature_block_end - feature_block_start
            true_templates[class_id,:cur_template_length,
                           feature_block_start:
                           feature_block_end
            ] = config_d['TRAINDATA']['template_means']
            if cur_template_length < true_templates.shape[1]:
                true_templates[class_id,cur_template_length:,
                               feature_block_start:
                               feature_block_end
                           ] = config_d['TRAINDATA']['background_means']

            
    elif config_d['TRAINDATA']['template_type'] == 'quadratic_curves':
        for i in xrange(len(true_templates)):
            if single_template_length:
                cur_template_length = template_lengths
            else:
                cur_template_length = template_lengths[class_id]

            for j in xrange(config_d['TRAINDATA']['num_curves']):
                true_templates[i,:cur_template_length] = np.maximum(
                    true_templates[i],
                    quadratic_curves_template(
                        cur_template_length,
                        config_d['TRAINDATA']['num_features'],
                        config_d['TRAINDATA']['template_means'],
                        config_d['TRAINDATA']['background_means'],
                        config_d['TRAINDATA']['template_blur_sigma']))

            if cur_template_length < true_templates.shape[1]:
                true_templates[i,cur_template_length:
                           ] = config_d['TRAINDATA']['background_means']

    else:
        print "no option corresponding to main.config"
        import pdb; pdb.set_trace()
        


    class_ids = np.random.randint(
        config_d['TRAINDATA']['num_classes'],
        size=config_d['TRAINDATA']['num_training']).astype(int)


    for i,x in enumerate(X):
        x[:] = (np.random.rand(config_d['TRAINDATA']['vector_length'],
                               config_d['TRAINDATA']['num_features']) < config_d['TRAINDATA']['background_means']).astype(np.uint8)

        if single_template_length:
            cur_template_length = template_lengths
        else:
            cur_template_length = template_lengths[class_id]

        x[start_times[i]:start_times[i]+cur_template_length] = (np.random.rand(cur_template_length,
                                                                                                       config_d['TRAINDATA']['num_features']) < true_templates[class_ids[i],:cur_template_length]).astype(np.uint8)


    if args.visualize_data is not None:
        for i,x in enumerate(X):
            plt.close('all')

            plt.imshow(
                x.T,
                origin='lower',
                interpolation='nearest',
                cmap='bone')
            plt.savefig("%s_%d.png" % (args.visualize_data,
                                       i),
                        bbox_inches='tight')
        
    if args.visualize_templates is not None:
        for c_id, c_template in enumerate(true_templates):
            plt.close('all')
            plt.imshow(c_template.T,
                       cmap='bone',
                       interpolation='nearest',
                       origin='lower',vmin=0,vmax=1)
            plt.axis('off')
            plt.savefig('%stemplates_%d.png' % (args.visualize_templates,
                                                c_id),
                        bbox_inches='tight')



    #import pdb; pdb.set_trace()

    np.save('%s_X.npy' % args.o,X)
    np.save('%s_start_times.npy' % args.o,start_times)
    np.save('%s_class_ids.npy' % args.o,class_ids)
    np.save('%s_true_templates.npy' % args.o, true_templates)

if __name__=="__main__":
    parser = argparse.ArgumentParser("""
""")
    parser.add_argument("-o",type=str,default=None,help="output file to save the examples to"
                        )
    parser.add_argument("-c",type=str,default='main.config',help="config file: default is config")
    parser.add_argument('-v',action='store_true',help='include if you want a print out about the examples as they are processed')
    parser.add_argument('--visualize_data',
                        type=str,
                        default=None,
                        help='default is None, in which case no visualization happens, otherwise a plot is saved to the path given as an argument')
    parser.add_argument('--visualize_templates',
                        type=str,
                        default=None,
                        help='default is None, in which case no visualization happens, otherwise a plot is saved to the path given as an argument')
    main(parser.parse_args())
