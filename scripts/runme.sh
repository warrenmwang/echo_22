#!/bin/bash 

LOGFILE="runme_log.txt"

{ time python ../warren-random/quantifying-performance/create_csv_ver_2.py -l Original_Pretrained_R2plus1DMotionSegNet.pth; } &>> $LOGFILE

echo "runme.sh script finished" &>> $LOGFILE

