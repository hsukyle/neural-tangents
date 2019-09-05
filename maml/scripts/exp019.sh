#!/usr/bin/env bash
visdom -p 8888 > /dev/null 2>&1 &
sleep 30
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
    --n_train_task $(expr 16 \* 5000) \
    --n_way 5 \
    --n_support 3 \
    --n_query 15 \
    --gradient_alignment_regularization 0.3 \
    --exp_name exp019
kill $!