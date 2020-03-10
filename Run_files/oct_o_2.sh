#$ -l tmem=8G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/CityScapes/projects_codes/Experiments

~/anaconda3/bin/python ours_v1_no_aug.py