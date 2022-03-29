# Introduction
This is the **official** PyToch implementation of **"Arbitrary-Shaped Scene Text Detection by Predicting Distance Map"**.  In this paper, a novel arbitrary-shaped text
detector based on distance map is proposed, which can flexibly and robustly detect arbitrary shaped texts via variable-width center lines.
# Installation
### Requirements:
Python3

PyTorch >= 1.2

cmake and GCC >= 4.9

**other:**

Download and compile opencv(3.x) source code.

**pck-config:** apt-get install pck-config

# Datasets
Download datasets from [Scene-Text-Detection](https://github.com/HCIILAB/Scene-Text-Detection).

**Note:** There are some conversion and reading tools in the **datasets** folder.

# Train
1. Copy the contents of the corresponding configuration file in the configs folder to config.py
2. Modify **trainroot** and **testroot** in config.py to the directories of training set and test set respectively, and modify **output_dir** to the directory where the model is saved
3. Execute **python train_ic15.py/train_ic17.py/train_ReCTS.py/train_curve.py/train_synth.py** to train
   
# Test
Execute **eval_ic15.py/eval_ReCTS.py/eval_curve.py** to test

# Result
**VGG16 in Paper([branch master](https://github.com/Whu-wxy/DistNet))**
dataset | pretrained on | precision | recall | F-measure
---- | --------- | --------- | ------ | ---------
IC17 | None | 75.6 |  61.9 | 68.1
IC15 | None | 87.9 |  78.6 | 83.0
IC15 | IC17 | 87.3 |  83.6 | 85.4
CTW1500 | None |  81.9 | 76.3 | 79.0
CTW1500 | IC17 | 84.0 | 77.2 | 80.4
Total-Text | None | 81.2 | 80.1 | 80.6
Total-Text | IC17 | 84.4 | 78.9 | 81.6

**DLA34 without pretrain**
dataset | precision | recall | F-measure
----  | --------- | ------ | ---------
Total-Text | 85.33 | 79.58 | 82.36
CTW1500 | 85.58 | 77.77 | 81.49
IC15 | 88.55 | 78.53 | 83.24

# Visualization
![Visualization](https://github.com/Whu-wxy/Text_Exp/blob/master/result.jpg "Visualization")

# Citation
Please cite the related works in your publications if it helps your research:

     @article{Wang2022ArbitraryshapedST,    
       title={Arbitrary-shaped scene text detection by predicting distance map},    
       author={Xinyu Wang and Yaohua Yi and Jibing Peng and Kaili Wang},   
       journal={Applied Intelligence},    
       year={2022}
     }

