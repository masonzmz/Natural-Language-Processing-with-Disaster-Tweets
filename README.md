# Natural-Language-Processing-with-Disaster-Tweets
Kaggle Competition: Predict which Tweets are about real disasters and which ones are not

## Final Rank and Result

**32 / 824 -- Top 3%** 
---
|Model|Best Accuracy|
|:---|:---|
|BERT|84.18%|
|RoBERTa|84.18%|


# Set-up
## Operation System:
![macOS Badge](https://img.shields.io/badge/-macOS-white?style=flat-square&logo=macOS&logoColor=000000) ![Linux Badge](https://img.shields.io/badge/-Linux-white?style=flat-square&logo=Linux&logoColor=FCC624) ![Ubuntu Badge](https://img.shields.io/badge/-Ubuntu-white?style=flat-square&logo=Ubuntu&logoColor=E95420)

## Library Requirements:
![Python](http://img.shields.io/badge/-3.8.13-eee?style=flat&logo=Python&logoColor=3776AB&label=Python) ![PyTorch](http://img.shields.io/badge/-1.12.0-eee?style=flat&logo=pytorch&logoColor=EE4C2C&label=PyTorch) ![Scikit-learn](http://img.shields.io/badge/-1.1.1-eee?style=flat&logo=scikit-learn&logoColor=e26d00&label=Scikit-Learn) ![NumPy](http://img.shields.io/badge/-1.22.3-eee?style=flat&logo=NumPy&logoColor=013243&label=NumPy) ![tqdm](http://img.shields.io/badge/-4.64.0-eee?style=flat&logo=tqdm&logoColor=FFC107&label=tqdm) ![pandas](http://img.shields.io/badge/-1.4.3-eee?style=flat&logo=pandas&logoColor=150458&label=pandas) ![SciPy](http://img.shields.io/badge/-1.8.1-eee?style=flat&logo=SciPy&logoColor=8CAAE6&label=SciPy) ![colorama](http://img.shields.io/badge/-0.4.5-eee?style=flat&label=colorama) ![cudatoolkit](http://img.shields.io/badge/-11.6.0-eee?style=flat&label=cudatoolkit) ![datasets](http://img.shields.io/badge/-2.4.0-eee?style=flat&label=datasets) ![matplotlib](http://img.shields.io/badge/-3.4.2-eee?style=flat&label=matplotlib) ![matplotlib](http://img.shields.io/badge/-3.7-eee?style=flat&label=nltk) ![tokenizers](http://img.shields.io/badge/-0.11.4-eee?style=flat&label=tokenizers) ![transformers](http://img.shields.io/badge/-4.18.0-eee?style=flat&label=transformers) ![seaborn](http://img.shields.io/badge/-0.11.2-eee?style=flat&label=seaborn)
## Environment
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
pip install transformers
pip install datasets
```

# Tuning Parameters
```
bash /src/run.sh
```

# Training
```
python3 /src/train.py \
--model_name [$model_name] \
--threshold [$threshold] \
--batchsize [$batchsize] \
--dropout [$dropout] \
--layer [$layer] 
```
