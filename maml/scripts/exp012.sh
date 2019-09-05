#!/usr/bin/env bash
visdom -p 8000 > /dev/null 2>&1 &
sleep 15
#python maml/ntk_maml.py \
#    --dataset sinusoid \
#    --n_hidden_layer 2 \
#    --n_hidden_unit 256 \
#    --bias_coef 1.0 \
#    --activation tanh \
#    --norm None \
#    --outer_step_size 1e-3 \
#    --outer_opt_alg adam \
#    --inner_opt_alg sgd \
#    --inner_step_size 0.5 \
#    --n_inner_step 1 \
#    --task_batch_size 16 \
#    --n_train_task $(expr 16 \* 10000) \
#    --n_way 1 \
#    --n_support 20 \
#    --n_query 20 \
#    --exp_name exp012 \
#    --debug
python maml/ntk_maml.py \
    --dataset omniglot \
    --n_hidden_layer 4 \
    --n_hidden_unit 64 \
    --bias_coef 1.0 \
    --activation relu \
    --norm batch_norm \
    --outer_step_size 1e-2 \
    --outer_opt_alg adam \
    --inner_opt_alg sgd \
    --inner_step_size 5 \
    --n_inner_step 1 \
    --task_batch_size 16 \
    --n_train_task $(expr 16 \* 10000) \
    --n_way 5 \
    --n_support 3 \
    --n_query 15 \
    --exp_name exp012
kill $!