#!/bin/bash -ex

# Copyright (c) 2013   (Author: Mark Stoehr)
# MIT


if [ $# -ne 1 ]; then
   echo "Argument should be the experiment directory--assumed to have a local/ and src/ directory containing the scripts and code for running the experiment, there should also be a "
   exit 1;
fi

dir=$1

conf=$dir/conf
mkdir -p $conf

# data is assumed to have length 37

# this template shaves off three time points on both
# ends of the data
# 1sh  -- indicates 1 shift
# 6c   -- indicates 6 classes
# 3st  -- indicates start time at 3
# 33l  -- indicates length 33 templates

start_time=3
num_shifts=1
num_classes=6
template_length=31
exp_suff=${num_shifts}sh_${num_classes}c_${start_time}st_${template_length}l
echo -e "[EMTRAINING]
tolerance=1e-6
num_shifts=${num_shifts}
num_classes=${num_classes}
start_time=${start_time}
template_length=${template_length}
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_uniform_shift'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=${start_time}
num_shifts=${num_shifts} " > $conf/b_p_${exp_suff}.config


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates


done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate

old_data_dir=/home/mark/Research/phoneclassification/data/local/data

local=$dir/local
for phn in b p ; do
    echo ${phn}
   `python $local/visualize_underlying_clusters.py \
       --data $old_data_dir/${phn}_train_examples_S.npy \
       --posteriors $exp/${phn}_${exp_suff}_posteriors.npy \
       --start_time ${start_time} \
       --template_length ${template_length} \
       -o $exp/${phn}_underlying_${exp_suff} --cmap hot`
done
##
# and now we repeat the same experiment but with a longer template
#
#

cat <<EOF > $conf/b_p_1sh_6c_0st_37l.config
[EMTRAINING]
tolerance=1e-6
num_shifts=1
num_classes=6
start_time=0
template_length=37
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_uniform_shift'
class_shift_type='unconstrained'
min_shift_sigma=.8


[INFERENCE]
start_time=0
num_shifts=1
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=1sh_6c_0st_37l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate

#
# we now repeat with shifts and see what happens
#
#

cat <<EOF > $conf/b_p_3sh_6c_0st_35l.config
[EMTRAINING]
tolerance=1e-6
num_shifts=3
num_classes=6
start_time=0
template_length=35
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=0
num_shifts=3
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=3sh_6c_0st_35l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate


#
# we now repeat withouts shifts but the full 35 length
# 
#

cat <<EOF > $conf/b_p_1sh_6c_1st_35l.config
[EMTRAINING]
tolerance=1e-6
num_shifts=1
num_classes=6
start_time=1
template_length=35
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=1
num_shifts=1
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=1sh_6c_1st_35l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate



#
# we now repeat with shifts and see what happens
#
#

start_time=0
num_shifts=7
num_classes=6
template_length=31
exp_suff=${num_shifts}sh_${num_classes}c_${start_time}st_${template_length}l
echo -e "[EMTRAINING]
tolerance=1e-6
num_shifts=${num_shifts}
num_classes=${num_classes}
start_time=${start_time}
template_length=${template_length}
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_uniform_shift'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=${start_time}
num_shifts=${num_shifts} " > $conf/b_p_${exp_suff}.config


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate

old_data_dir=/home/mark/Research/phoneclassification/data/local/data

local=$dir/local
for phn in b p ; do
    echo ${phn}
   `python $local/visualize_underlying_clusters.py \
       --data $old_data_dir/${phn}_train_examples_S.npy \
       --posteriors $exp/${phn}_${exp_suff}_posteriors.npy \
       --start_time ${start_time} \
       --template_length ${template_length} \
       -o $exp/${phn}_underlying_${exp_suff} --cmap hot`
done


#
# and now we see what happens if we initialize with the previous
# templates that had not shifts

start_time=0
num_shifts=7
num_classes=6
template_length=31
exp_suff=${num_shifts}sh_${num_classes}c_${start_time}st_${template_length}l_previnit
echo -e "[EMTRAINING]
tolerance=1e-6
num_shifts=${num_shifts}
num_classes=${num_classes}
start_time=${start_time}
template_length=${template_length}
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='template'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=${start_time}
num_shifts=${num_shifts} " > $conf/b_p_${exp_suff}.config


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
init_suff=1sh_6c_3st_31l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --init_templates $exp/${phn}_${init_suff}_templates.npy\
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate

old_data_dir=/home/mark/Research/phoneclassification/data/local/data

local=$dir/local
for phn in b p ; do
    echo ${phn}
   `python $local/visualize_underlying_clusters.py \
       --data $old_data_dir/${phn}_train_examples_S.npy \
       --posteriors $exp/${phn}_${exp_suff}_posteriors.npy \
       --start_time ${start_time} \
       --template_length ${template_length} \
       -o $exp/${phn}_underlying_${exp_suff} --cmap hot`
done


#
# now we use the same shifts except that
# we use a normal distribution over the shifts
# and we use the same distribution for all the shifts

cat <<EOF > $conf/b_p_7sh_6c_0st_31l_normal_independent.config
[EMTRAINING]
tolerance=1e-6
num_shifts=7
num_classes=6
start_time=0
template_length=31
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'
class_shift_type='normal_independent'
min_shift_sigma=.1

[INFERENCE]
start_time=0
num_shifts=7
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=7sh_6c_0st_31l_normal_independent
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate

#
# now we use the same shifts except that
# we use a normal distribution over the shifts
# and we use a different distribution for all the shifts depending on class

cat <<EOF > $conf/b_p_7sh_6c_0st_31l_normal_dependent.config
[EMTRAINING]
tolerance=1e-6
num_shifts=7
num_classes=6
start_time=0
template_length=31
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'
class_shift_type='normal_dependent'
min_shift_sigma=.1

[INFERENCE]
start_time=0
num_shifts=7
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=7sh_6c_0st_31l_normal_dependent
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate



#
# we now repeat the same experiment but where we train with
# zero shift in the initialization
#

cat <<EOF > $conf/main.config
[TRAINDATA]
num_training=100
template_lengths=(10,20)
template_means=.9
background_means=.2
vector_length=40
num_features=16
num_shifts=9
random_seed=0
num_classes=2
template_type='uniform'

[EMTRAINING]
tolerance=1e-6
num_shifts=11
num_classes=2
start_time=0
template_length=20
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'

[INFERENCE]
start_time=0
num_shifts=1
EOF

exp=$dir/shift_class_lengths_exp
mkdir -p $exp
$src/bernoullishiftonly_em.py -c $conf/main.config -i $data/generated_X.npy \
    --out_templates $exp/out_shift_spike_templates.npy\
    --out_background $exp/out_shift_spike_background.npy\
    --out_posteriors $exp/out_shift_spike_posteriors.npy\
    --out_class_shift_probs $exp/out_shift_spike_class_shift_probs.npy\
    -o $exp/out_shift_spike_  --visualize_templates

#
# and now we see what happens if we initialize with the previous
# templates that had not shifts

cat <<EOF > $conf/b_p_5sh_6c_1st_31l_previnit.config
[EMTRAINING]
tolerance=1e-6
num_shifts=5
num_classes=6
start_time=1
template_length=31
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='template'
class_shift_type='unconstrained'
min_shift_sigma=.8

[INFERENCE]
start_time=1
num_shifts=5
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=5sh_6c_1st_31l_previnit
init_suff=1sh_6c_3st_31l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --init_templates $exp/${phn}_${init_suff}_templates.npy\
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate



#
# and now we see what happens if we initialize with the previous
# templates that had not shifts

cat <<EOF > $conf/b_p_5sh_6c_1st_31l_previnit_ni.config
[EMTRAINING]
tolerance=1e-6
num_shifts=5
num_classes=6
start_time=1
template_length=31
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='template'
class_shift_type='normal_independent'
min_shift_sigma=.1


[INFERENCE]
start_time=1
num_shifts=5
EOF


src=$dir/src
local=$dir/local
data=$dir/data
mkdir -p $data

exp=$dir/exp/p_b_exp
mkdir -p $exp
exp_suff=5sh_6c_1st_31l_previnit_ni
init_suff=1sh_6c_3st_31l
for phn in b p ; do
    $src/bernoullishiftonly_em.py -c $conf/b_p_${exp_suff}.config -i $data/${phn}_train.npy \
    --init_templates $exp/${phn}_${init_suff}_templates.npy\
    --out_templates $exp/${phn}_${exp_suff}_templates.npy\
    --out_background $exp/${phn}_${exp_suff}_background.npy\
    --out_posteriors $exp/${phn}_${exp_suff}_posteriors.npy\
    --out_class_shift_probs $exp/${phn}_${exp_suff}_class_shift_probs.npy\
    -o $exp/${phn}_${exp_suff}_  --visualize_templates

    
done

$src/bernoullishift_binary_classify.py -c $conf/b_p_${exp_suff}.config \
        --models $exp/b_${exp_suff}_templates.npy\
                 $exp/p_${exp_suff}_templates.npy\
        --data data/b_dev.npy data/p_dev.npy\
        --out $exp/b_p_${exp_suff}_results_\
        --bgds $exp/b_${exp_suff}_background.npy\
               $exp/p_${exp_suff}_background.npy | awk '{ print $3 }' >$exp/b_p_${exp_suff}.error_rate




#
# and then we do the same training without any shifts
#
#

cat <<EOF > $conf/main.config
[TRAINDATA]
num_training=100
template_lengths=(10,20)
template_means=.9
background_means=.2
vector_length=40
num_features=16
num_shifts=9
random_seed=0
num_classes=2
template_type='uniform'

[EMTRAINING]
tolerance=1e-6
num_shifts=1
num_classes=2
start_time=0
template_length=20
min_prob=.01
class_shift_min_prob=.005
random_seed=0
initialization='random_class_zero_shift'

[INFERENCE]
start_time=0
num_shifts=1
EOF

exp=$dir/shift_class_lengths_exp
mkdir -p $exp
$src/bernoullishiftonly_em.py -c $conf/main.config -i $data/generated_X.npy \
    --out_templates $exp/out_no_shift_templates.npy\
    --out_background $exp/out_no_shift_background.npy\
    --out_posteriors $exp/out_no_shift_posteriors.npy\
    --out_class_shift_probs $exp/out_no_shift_class_shift_probs.npy\
    -o $exp/out_no_shift_  --visualize_templates

