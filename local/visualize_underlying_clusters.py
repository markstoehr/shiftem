import numpy as np
import argparse
from src import bernoullishiftonly_em
import matplotlib.pyplot as plt

def main(args):
    """
    Visualize the templates with the underlying data
    """
    data = np.load(args.data)
    posteriors = np.load(args.posteriors)
    
    Z_templates = bernoullishiftonly_em.cluster_underlying_data(data,posteriors,args.start_time,args.template_length)

    min_val = Z_templates.min()
    max_val = Z_templates.max()
    for c_id, c_template in enumerate(Z_templates):
        plt.close('all')
        plt.imshow(c_template.T,
                   cmap=args.cmap,
                   interpolation='nearest',
                   origin='lower',
                   vmin=min_val,vmax=max_val)
        plt.axis('off')
        plt.savefig('%s_templates_%d.png' % (args.o,c_id),
                    bbox_inches='tight')

if __name__=="__main__":
    parser = argparse.ArgumentParser("""Cluster the underlying data
    sample run:

    python $local/visualize_underlying_clusters.py\ 
       --data $old_data_dir/${phn}_train_examples_S.npy\
       --posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
       --start_time ${start_time}\
       --template_length ${template_length}\
       -o $exp/${phn}_underlying

    the posteriors computed over the edge data""")
    parser.add_argument("--data",type=str,
                        help='path to where the underlying data (genernally spectrogram features) are saved--should be a numpy archive .npy')
    parser.add_argument('--posteriors',type=str,
                        help='path to where the posteriors are saved')
    parser.add_argument('--start_time',type=int,
                        help='start time for the beginning of the window over which the shiftable template is used to model the data')
    parser.add_argument('--template_length',type=int,
                        help='length of the template model')
    parser.add_argument('-o',type=str,help='prefix to string where things are going to be saved')
    parser.add_argument('--cmap',type=str,default='bone',help='color map name to use for the plot')
    main(parser.parse_args())
    
