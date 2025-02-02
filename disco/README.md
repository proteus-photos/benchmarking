# General
This is the official code for DISCO: Adversarial Defense with Local Implicit Functions, accepted by NeurIPS 2022

# Evironment
Creating the environment
```
conda env create -f environment.yml 
conda activate disco
cd disco
pip install --upgrade git+https://github.com/BorealisAI/advertorch.git
pip install --upgrade git+https://github.com/fra31/auto-attack.git@6482e4d6fbeeb51ae9585c41b16d50d14576aadc#egg=autoattack
cd testing
python setup.py install
cd ../
source install.sh
```

# Data Preparation
All the data preparation code are inside the data_preparation folder.
Follow the instruction to create clean adv training/testing pairs. By default, we use training pairs created using pgd. 
```
cd data_preparation
```
## Cifar10
Please read the shell script for more instruction. The dataset will downloaded automatically. 
```
cd cifar10
sh create_dataset.sh
```

## Cifar100
Please read the shell script for more instruction. Download the pretrained Cifar100 WideResNet_28_10 classifier from [google drive](https://drive.google.com/file/d/1C4iyTTeD0psuMzryvkK7o8xBO0YTff1q/view?usp=sharing) to data_preparation/cifar100/WideResNet_28_10_best.pth.tar

The dataset will downloaded automatically. 
```
cd cifar100
sh create_dataset.sh
```

## ImageNet
Please read the shell script for more instruction. The imagenet train/imagenet val contains the imagenet training/val set, organized by the imagenet class (e.g. n04252225, n04482393, n07714990)  
```
cd imagenet
sh create_dataset.sh
```

# Disco Training
```
cd training
```
All the disco training code are inside the training folder. Download the pretrained checkpoint from [google drive](https://drive.google.com/file/d/1kMjqPVmu1xNhmZdPf25BN9JQ0j9LBar5/view?usp=sharing) and upzip it to the "training/save" folder. The below training code for each dataset in debug mode. Remove the --debug flag for full training.

## Cifar10
The config files for training disco on cifar10 is stored under configs/cifar10. 
```
CUDA_VISIBLE_DEVICES=0 sh scripts/train_disco_cifar10.sh
```
## Cifar100
The config files for training disco on cifar100 is stored under configs/cifar100. 
```
CUDA_VISIBLE_DEVICES=0 sh scripts/train_disco_cifar100.sh
```
## ImageNet
The config files for training disco on imagenet is stored under configs/imagenet. 
```
CUDA_VISIBLE_DEVICES=0 sh scripts/train_disco_imagenet.sh
```


# Disco Testing
```
cd testing
```
## Download Pretrained Classifiers
Download Cifar10 pretrained classifiers from [google drive](https://drive.google.com/file/d/1gJjRKU09diktm_Q1jw-Tg_lAmGV73kPo/view?usp=share_link) unzip to testing/cifar10_models/state_dicts. Download Cifar100 pretrained classifiers from [google drive](https://drive.google.com/file/d/1UuM-dcHxPpUjXYncAUGwafDZVp7mXqpP/view?usp=sharing) unzip to testing/WideResNet-pytorch/runs. Link the testing/data/val to the folder that contains imagenet validation folder, organized by the imagenet class (e.g. n04252225, n04482393, n07714990)

## Cifar10
The scripts for testing disco on cifar10 is stored under script/cifar10. Please refer to each shell script for more instruction and add debug flag to run under debug mode. To run all cifar10 experiments, run
```
sh run_cifar10.sh
```


### Oblivious Setting
Evaluate transformation based and adversarially trained based baselines
```
sh scripts/cifar10/oblivious_setting/robustbench_baselines.sh debug
sh scripts/cifar10/oblivious_setting/robustbench_baselines.sh
```

Evaluate DISCO
```
sh scripts/cifar10/oblivious_setting/robustbench_disco.sh debug
sh scripts/cifar10/oblivious_setting/robustbench_disco.sh
```

Measure the transfer performance for different datasets and classifiers
```
sh scripts/cifar10/oblivious_setting/dataset_classfier_transfer.sh debug
sh scripts/cifar10/oblivious_setting/dataset_classfier_transfer.sh
```

Measure the defense transfer performance for different attacks
```
sh scripts/cifar10/oblivious_setting/defense_transfer.sh debug
sh scripts/cifar10/oblivious_setting/defense_transfer.sh
```

Improving different robust classifiers with DISCO
```
sh scripts/cifar10/oblivious_setting/improve_robust_classifier.sh debug
sh scripts/cifar10/oblivious_setting/improve_robust_classifier.sh
```

Measure the performance under bpda attack 
```
sh scripts/cifar10/oblivious_setting/bpda.sh debug
sh scripts/cifar10/oblivious_setting/bpda.sh
```

### Adaptive Setting
Measure the performance under bpda attack 
```
sh scripts/cifar10/adaptive_setting/bpda.sh debug
sh scripts/cifar10/adaptive_setting/bpda.sh
```

Measure the speed of bpda attack for both cascade disco and disco 
```
sh scripts/cifar10/adaptive_setting/bpda_speed.sh debug
sh scripts/cifar10/adaptive_setting/bpda_speed.sh
```



## Cifar100
The scripts for testing disco on cifar100 is stored under script/cifar100. Please refer to each shell script for more instruction and add debug flag to run under debug mode. To run all cifar100 experiments, run
```
sh run_cifar100.sh
```


### Oblivious Setting
Evaluate transformation based and adversarially trained based baselines
```
sh scripts/cifar100/oblivious_setting/robustbench_baselines.sh debug
sh scripts/cifar100/oblivious_setting/robustbench_baselines.sh
```

Evaluate DISCO
```
sh scripts/cifar100/oblivious_setting/robustbench_disco.sh debug
sh scripts/cifar100/oblivious_setting/robustbench_disco.sh
```

Improving different robust classifiers with DISCO
```
sh scripts/cifar100/oblivious_setting/improve_robust_classifier.sh debug
sh scripts/cifar100/oblivious_setting/improve_robust_classifier.sh
```

Measure the defense transfer performance for different attacks
```
sh scripts/cifar100/oblivious_setting/defense_transfer.sh debug
sh scripts/cifar100/oblivious_setting/defense_transfer.sh
```

Measure the transfer performance for different classifiers
```
sh scripts/cifar100/oblivious_setting/classfier_transfer.sh debug
sh scripts/cifar100/oblivious_setting/classfier_transfer.sh
```

## ImageNet
The scripts for testing disco on imagenet is stored under script/imagenet. Please refer to each shell script for more instruction and add debug flag to run under debug mode. To run all imagenet experiments, run
```  
sh run_imagenet.sh                                                                                                                                                         
```  

Evaluate transformation based and adversarially trained based baselines
```
sh scripts/imagenet/oblivious_setting/robustbench_baselines.sh debug
sh scripts/imagenet/oblivious_setting/robustbench_baselines.sh
```

Evaluate DISCO
```
sh scripts/imagenet/oblivious_setting/robustbench_disco.sh debug
sh scripts/imagenet/oblivious_setting/robustbench_disco.sh
```

Improving different robust classifiers with DISCO
```
sh scripts/imagenet/oblivious_setting/improve_robust_classifier.sh debug
sh scripts/imagenet/oblivious_setting/improve_robust_classifier.sh
```

Measure the defense transfer performance for different attacks
```
sh scripts/imagenet/oblivious_setting/defense_transfer.sh debug
sh scripts/imagenet/oblivious_setting/defense_transfer.sh
```

Measure the transfer performance for different classifiers
```
sh scripts/imagenet/oblivious_setting/classfier_transfer.sh debug
sh scripts/imagenet/oblivious_setting/classfier_transfer.sh
```


## Citation
If you find this method useful in your research, please cite this article:
```
@inproceedings{
ho2022disco,
title={{DISCO}: Adversarial Defense with Local Implicit Functions},
author={Chih-Hui Ho and Nuno Vasconcelos},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=vgIz0emVTAd}
}
```

# Acknowledgement
Please email to Chih-Hui (John) Ho (chh279@eng.ucsd.edu) if further issues are encountered. We used the code from 
1. https://github.com/yinboc/liif
2. https://github.com/RobustBench/robustbench
3. https://github.com/BorealisAI/advertorch
4. https://github.com/thu-ml/ares

