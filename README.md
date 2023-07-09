# image-similarity  🪞

That is our attempt on solving the CSC Hackathon 2023 problem provided by LUN.UA

## Results  🚀
|      Model      |  Fscore  |
| :-------------: | :-------: |
|    `ResNet152`  |  `0.997`  |


## Brief model desciption 🔍
We take [ResNet 152](https://arxiv.org/abs/1512.03385) and retrain the last fully connected layer.
Then we produce embeddings of images using our network and measure distance between them.
Read more about Contrastive loss and Siamese Network if interested or check the implementation.

Model was trained of **32000** pairs of images on `MacBook Pro 2021, M1, 32GB`.
Evaluating the test dataest consisting of **22660** images took **1063 seconds**, or **0.046 seconds per pair**.

## Testing 
Feel free to experiment with the model yourself! 

Check out `ResNet.ipynb`
