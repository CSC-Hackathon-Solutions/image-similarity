# image-similarity  🪞

That is our attempt on solving the CSC Hackathon 2023 problem provided by LUN.UA

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
Read more about Contrastive loss and Siamese Network if interested or check the implementation.

Model was trained of **32000** pairs of images on `MacBook Pro 2021, M1, 32GB`.
Evaluating the test dataest consisting of **22660** images took **1063 seconds**, or **0.046 seconds per pair**.

## Testing 
Feel free to experiment with the model yourself! 

Check out `ResNet.ipynb`
