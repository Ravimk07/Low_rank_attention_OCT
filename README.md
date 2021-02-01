# MICCAI_2020_OCT
Paper link: https://www.medrxiv.org/content/10.1101/2020.08.13.20174250v2

## How to use this repo with your own datasets:
1. Prepare your data and save them as pictures in .tiff

2. Customize your dataset into this structure:

-- data_directory
   -- train
      -- images
      -- masks
   -- val
      -- images
      -- masks
   -- test
      -- images
      -- masks

3. Change data path in Run.py

4. Fine tune your models, change hyper-parameters in Run.py

Contact: moucheng.xu.18@ucl.ac.uk 

## Citation:
If you find the code or the README useful for your research, please cite our paper:
```
@article{OCTseg,
  title={Bio-Inspired Attentive Segmentation of Retinal OCT imaging},
  author={Xu*, Mou-Cheng and Lazaridis*, Georgios and S. Afgeh, Saman and Garway-Heath, David},
  journal={OMIA workshop at MICCAI},
  year={2020},
}
```
