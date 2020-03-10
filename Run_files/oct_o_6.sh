#$ -l tmem=4G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/CityScapes/projects_codes/MICCAI_2020_OCT

~/anaconda3/bin/python ours_v1_ultra_light.py