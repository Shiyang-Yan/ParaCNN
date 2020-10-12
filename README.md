Requirements

Python 3.7 (need COCO evaluation package for Python 3)

PyTorch 1.2.1 (with torchvision)

cider

coco-caption

spacy (to tokenize words)

h5py (to store features)

scikit-image (to process images)

Procedure:

1. Download the data from the following link, put it in the directory. https://drive.google.com/drive/folders/1oFImRV4pyZGjbZC76fT7pmbhLLiNvK22?usp=sharing
2. Training: run the 'python train_twin_wgan.py', and change the path accordingly.
3  Testing： run the 'python val_att.py', and change the path accordingly.
