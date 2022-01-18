# DLCV Final Project ( Food-classification-Challenge )



# How to run your code?
> TODO: Please provide example scripts to run your code. For example, 
> 1. python3 preprocessing.py <Path to Blood_data>
> 2. python3 inference.py <Path to the output csv file>
### Dataset
    bash ./get_dataset.sh
## Install packages
    pip install -r requirements.txt
## TransFG (Baseline)
    cd TransFG/
- Install apex
```
cd apex/
pip install -v --no-cache-dir ./
cd ..
```
- estimated training hours
```
15 hours on four GTX-1080Ti GPU.
```
- Training
```
bash download_vit.sh
bash train.sh $1 $2
$1 is the directory of the food dataset (e.g. ./food_data)
$2 is the directory to store the checkpoint (e.g. ./output)
```
- Download checkpoint
```
bash download_baseline.sh
```
- Inference
```
bash test_main_track.sh $1 $2 $3
$1 is the sample submission file (e.g. sample_submission_main_track.csv)
$2 is the path of output csv file (e.g. output.csv)
$3 is the path of model chekpoint (e.g. ./model.bin)
```
## Multi-task
    cd TransFG_Multitask/
- Training
```
bash ./train.sh $1 $2 $3 $4
$1 is the directory of the food dataset (e.g. ./food_data) 
$2 is the path of pretrained model checkpoint (e.g. ./pretrained.bin)
$3 is the directory to store the checkpoint (e.g. ./output)
$4 is the output name of the checkpoint (e.g. multitask)
```
- Download checkpoint
```
bash ./download.sh
```
- Inference
```
bash ./test.sh $1 $2 $3
$1 is the sample submission file (e.g. sample_submission_main_track.csv)
$2 is the path of output csv file (e.g. output.csv)
$3 is the path of model chekpoint (e.g. ./pretrained.bin)
```

