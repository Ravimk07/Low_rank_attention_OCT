#$ -l tmem=6G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/CityScapes/projects_codes/MICCAI_2020_OCT

~/anaconda3/bin/python ours_v1_light_2.py