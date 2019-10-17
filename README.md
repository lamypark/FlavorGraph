# FlavorNet 2.0
This repository provides a Pytorch implementation of **KitcheNette**, Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. **KitcheNette** is able to predict and recommend complementary and novel food ingredients pairings at the same time.

> **KitcheNette: Predicting and Ranking Food Ingredient Pairings using Siamese Neural Networks** <br>
> *Donghyeon Park\*, Keonwoo Kim, Yonggyu Park, Jungwoon Shin and Jaewoo Kang* <br>
> *Accepted and to be appear in IJCAI-2019* <br><br>
> *Our paper is available at:* <br>
> *https://www.ijcai.org/proceedings/2019/822* <br><br>
> You can try our demo version of **KitchenNette**: <br>
> *http://kitchenette.korea.ac.kr/*
> 
> For more details to find out what we do, please visit *https://dmis.korea.ac.kr/*

## Pipeline & Abstract
![figure](/data/figure_together.png)
<p align="center">
  <b> The Concept of KitcheNette (Left) & KitcheNette Model Architecture (Right) </b>
</p>

**Abstract** <br>
As a vast number of ingredients exist in the culinary world, there are countless food ingredient pairings, but only a small number of pairings have been adopted by chefs and studied by food researchers. In this work, we propose KitcheNette which is a model that predicts food ingredient pairing scores and recommends optimal ingredient pairings. KitcheNette employs Siamese neural networks and is trained on our annotated dataset containing 300K scores of pairings generated from numerous ingredients in food recipes. As the results demonstrate, our model not only outperforms other baseline models but also can recommend complementary food pairings and discover novel ingredient pairings.

## Prerequisites & Development Environment
- Python 3.6
- PyTorch 0.4.0
- Numpy (>=1.12)
- Maybe there are more. If you get an error, please try `pip install "pacakge_name"`.

- CUDA 9.0
- Tested on NVIDIA GeForce Titan X Pascal 12GB

## Dataset
- **[kitchenette_pairing_scores.csv](https://drive.google.com/file/d/1hX7L3UZUVspNHCjDbgCjuI5niQlBXXMh/view?usp=sharing) (78MB)** <br>
You can download and see our 300k food ingredient pairing scores defined on NPMI.

- **\[For Training\] [kitchenette_dataset.pkl](https://drive.google.com/file/d/1tUbwr7COW0lkiGkM3gafeGwtQncWd8wC/view?usp=sharing) (49MB)** <br>
For your own training, download our pre-processed dataset and place it in `data` folder. <br>
This pre-processed dataset 1) contains all the input embeddings, 2) is split into train[8]:valid[1]:test[2], and 3) and each split is divided into mini-batches for efficent training.

## Training & Test
```
python3 main.py --data-path './data/kitchenette_dataset.pkl'
```
## Prediction for *Unknown* Pairings
You need the following three files to predict *unknown* pairings
- **[kitchenette_pretrained.mdl](https://drive.google.com/file/d/1y5lFnECVdAaEikezeYipIABo4-5gvcbb/view?usp=sharing) (79MB)** <br>
Download our pre-trained model for prediction of *unknown* pairings and place it in `results` folder. <br>
or you can predict the pairing with your own model by substituting the model file. <br>

- **[kitchenette_unknown_pairings.csv](https://drive.google.com/file/d/10NECr9NAZ1tuZroJVVY4DmZY9Ox7vOyM/view?usp=sharing) (308KB)** <br>
Download the sample unknown pairings and place it in `data` folder. <br>
This files contains approximately 5,000 pairings that have no scores because that they are ralely or never used togeter. You can edit this file to score any pair of two ingredeints that you would like to find out.

- **[kitchenette_embeddings.pkl](https://drive.google.com/file/d/1cFRfrAEWqltQyLcALa1wwjQL7ssGK6ZD/view?usp=sharing) (8MB)** <br>
Download the sample ingredient embeddings for exisiting ingredients and place it in `data` folder. <br>
For this version, unfortunately, our model only scores the ingredients with pre-traiend embeddings.

```
python3 main.py --save-prediction-unknowns True \
                --model-name 'kitchenette_pretrained.mdl' \
                --unknown-path './data/kitchenette_unknown_pairings.csv' \
                --embed-path './data/kitchenette_embeddings.pkl' \
                --data-path './data/kitchenette_dataset.pkl'
```

## Contributors
**Donghyeon Park, Keonwoo Kim** <br>
DMIS Labatory, Korea University, Seoul, South Korea <br>
Please, report bugs and missing info to Donghyeon `parkdh (at) korea.ac.kr`.

## Citation
```
@article{park2019kitchenette,
  title={KitcheNette: Predicting and Ranking Food Ingredient Pairings using Siamese Neural Networks},
  author={Park, Donghyeon and Kim, Keonwoo and Park, Yonggyu and Shin, Jungwoon and Kang, Jaewoo},
  journal={Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence},
  year={2019}
}
```

## Liscense
Apache License 2.0
