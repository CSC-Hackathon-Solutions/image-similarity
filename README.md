# near-duplicate images detection  🪞

That is our attempt on solving the CSC Hackathon 2023 [problem provided by LUN.UA](https://www.kaggle.com/competitions/csc-hackathon-2023-lunua-task)

## Results  🚀
```
training was done on 32000 samples

              precision    recall  f1-score   support
           0    0.99825   0.99935   0.99880     15431
           1    0.99861   0.99627   0.99744      7229
    accuracy                        0.99837     22660
   macro avg    0.99843   0.99781   0.99812     22660
weighted avg    0.99837   0.99837   0.99837     22660
```


## Brief model desciption 🔍
We take [ResNet 152](https://arxiv.org/abs/1512.03385) and retrain the last fully connected layer.
Then we produce embeddings of images using our network and measure distance between them.
Read more about Contrastive loss and Siamese Network if you are interested or check the implementation.

Model was trained of **32000** pairs of images on `MacBook Pro 2021, M1, 32GB` (took ~25 minutes).
Evaluating the test dataest consisting of **22660** images took **1063 seconds**, or **0.046 seconds per pair**.

## Testing 
Feel free to experiment with the model yourself! 

Check out `ResNet.ipynb`
Don't forget to load the unzipped photoes to the folder `data/images` though.

## Directory

```

image-similarity
  |
  | - Data (metadata files for the test and training data)
  |
  | - ResNet.ipynb (train and experiment with the model yourself)
  |
  | - hypertuner.ipynb (optuna hyperparameter tuner)
  |
  | - hypertuner-augment.ipynb (modified optuna hyperparameter tuner that uses data augmentations)
  |
  | - utils.py (miscellanious methods and classes: custom Dataset, Dataloader, model train, etc)
  |
    . . .  (you won't need the other ones most likely. mostly experiments and scripts for data loading)
 
```
