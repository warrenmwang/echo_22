#!/bin/bash

LOGFILE="run_log.txt"

{ time python train_new_model_2.py --name dropout_v3_0_25_R2plus1D_18_MotionNet.pth; } &>> $LOGFILE

echo "fin" >> $LOGFILE