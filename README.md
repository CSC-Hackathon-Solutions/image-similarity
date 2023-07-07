# image-similarity 🪞

That is our attempt on solving the CSC Hackathon 2023 problem provided by LUN.UA

## Results 🚀
|      Model      | Fscore  |
| :-------------: | :-------: |
| `Threshold`     |  `0.971`  |
| `ResNet50 3k batches, fc no tweaks` |  `0.98717`  |


## Brief model desciption 🔍
We take (ResNet 152)[https://arxiv.org/abs/1512.03385] and retrain the last fully connected layer.
Then we produce embeddings of images using our network and using contrastive loss / siamese network concepts train it.
TODO sth else ???

## Feel free to experiment with the model yourself! check out main.ipynb