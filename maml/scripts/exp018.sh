#!/usr/bin/env bash
python maml/ntk_maml.py \
    --dataset sinusoid \
    --n_hidden_layer 2 \
    --n_hidden_unit 256 \
    --bias_coef 1.0 \
    --activation tanh \
    --norm None \
    --outer_step_size 1e-3 \
    --outer_opt_alg adam \
    --inner_opt_alg sgd \
    --inner_step_size 0.5 \
    --n_inner_step 1 \
    --task_batch_size 16 \
    --n_train_task $(expr 16 \* 10000) \
    --n_way 1 \
    --n_support 20 \
    --n_query 20 \
    --noise_std 0 \
    --gradient_alignment_regularization 0.1  \
    --exp_name exp018 
#    --debug