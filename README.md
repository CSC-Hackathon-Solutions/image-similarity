# image-similarity  🪞

That is our attempt on solving the CSC Hackathon 2023 problem provided by LUN.UA

## Results  🚀
|      Model      |  Fscore  |
| :-------------: | :-------: |
|    `ResNet152`  |  `0.997`  |


## Brief model desciption 🔍
We take [ResNet 152](https://arxiv.org/abs/1512.03385) and retrain the last fully connected layer.
Then we produce embeddings of images using our network and using contrastive loss / siamese network concepts train it.
TODO sth else ???

## Testing 
Feel free to experiment with the model yourself! Check out ResNet.ipynb
