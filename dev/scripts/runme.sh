#!/bin/bash 

LOGFILE="runme_log.txt"

echo "orig" >> $LOGFILE
{ time python ../quantifying-performance/create_csv_ver_3.py -l Original_Pretrained_R2plus1DMotionSegNet.pth; } &>> $LOGFILE

echo "drop v3" >> $LOGFILE
{ time python ../quantifying-performance/create_csv_ver_3.py -l dropout_v3_0_10_R2plus1DMotionSegNet.pth; } &>> $LOGFILE


echo "runme.sh script finished" &>> $LOGFILE