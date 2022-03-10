#!/usr/bin/env bash
EXP_NAME=$1
python train.py $EXP_NAME \
--secret_size 100 \
--num_steps 150000 \
--rnd_trans .1 \
--rnd_trans_ramp 10000 \
--secret_loss_scale 2 \
--secret_loss_ramp 1 \
--l2_loss_scale 1.5 \
--l2_loss_ramp 15000 \
--text_loss_scale 1.5 \
--text_loss_ramp 15000 \
--factor 1.0 \
--y_scale 100 \
--u_scale 1 \
--v_scale 1 \
--r_scale 3 \
--g_scale 6 \
--b_scale 1
