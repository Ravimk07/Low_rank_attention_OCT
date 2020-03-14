#$ -l tmem=8G
#$ -l gpu=true
#$ -S /bin/bash
#$ -j y
#$ -wd /cluster/project0/CityScapes/projects_codes/MICCAI_2020_OCT/Experiments

~/anaconda3/bin/python ours_large_kernel.py