# DLCV Final Project ( Food-classification-Challenge )

# How to run your code?
> TODO: Please provide example scripts to run your code. For example, 
> 1. python3 preprocessing.py <Path to Blood_data>
> 2. python3 inference.py <Path to the output csv file>
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

    
# Usage
To start working on this final project, you should clone this repository into your local machine by using the following command:

    git clone https://github.com/DLCV-Fall-2021/final-project-challenge-3-<team_name>.git
Note that you should replace `<team_name>` with your own team name.

For more details, please click [this link](https://drive.google.com/drive/folders/13PQuQv4dllmdlA7lJNiLDiZ7gOxge2oJ?usp=sharing) to view the slides of Final Project - Food image classification. **Note that video and introduction pdf files for final project can be accessed in your NTU COOL.**

### Dataset
In the starter code of this repository, we have provided a shell script for downloading and extracting the dataset for this assignment. For Linux users, simply use the following command.

    bash ./get_dataset.sh
The shell script will automatically download the dataset and store the data in a folder called `food_data`. Note that this command by default only works on Linux. If you are using other operating systems, you should download the dataset from [this link](https://drive.google.com/file/d/1IYWPK8h9FWyo0p4-SCAatLGy0l5omQaw/view?usp=sharing) and unzip the compressed file manually.

> ⚠️ ***IMPORTANT NOTE*** ⚠️  
> 1. Please do not disclose the dataset! Also, do not upload your get_dataset.sh to your (public) Github.
> 2. You should keep a copy of the dataset only in your local machine. **DO NOT** upload the dataset to this remote repository. If you extract the dataset manually, be sure to put them in a folder called `food_data` under the root directory of your local repository so that it will be included in the default `.gitignore` file.

> 🆕 ***NOTE***  
> For the sake of conformity, please use the `python3` command to call your `.py` files in all your shell scripts. Do not use `python` or other aliases, otherwise your commands may fail in our autograding scripts.

# Q&A
If you have any problems related to Final Project, you may
- Use TA hours
- Contact TAs by e-mail ([ntudlcv@gmail.com](mailto:ntudlcv@gmail.com))
- Post your question under Final Project FAQ section in NTU Cool Discussion
