#!/bin/bash

# script to train new models, if you wish to train a new model with different structure
# than the original R2plus1D_18_MotionNet, you will need a new class definition of the model
# inheriting from nn.Module
# make those changes and then provide the name of that new class definition
# below with the file extension .pth appended

# model gets saved to tmp_save_models/{model_name}"
# training log gets saved to training_logs/

MODEL="dropout_v3_0_10_R2plus1D_18_MotionNet.pth"
LOGFILE="./training_logs/${MODEL}_training_log.txt"

(time python train_new_model_2.py --name $MODEL) 2>&1 | tee -a $LOGFILE