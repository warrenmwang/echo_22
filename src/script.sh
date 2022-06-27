#!/bin/bash
for i in $(ls); do 
	if [ -d $i  ]; then
		echo $i is a directory;
	else
		if ["$i" = "script.sh" ]; then 
			:
		else
			echo $i is a file;
			cat $i | grep -w 'VolumeTracings.csv';
		fi
	fi
done 

