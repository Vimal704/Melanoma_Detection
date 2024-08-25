Implementation of paper: [Deep residual network with regularised fisher framework for detection of Melanoma](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cvi.2018.5238)

The Paper proposes to extract features from pretrained on Imagenet, finetuned on ISBI dataset ResNet50 model, 
further Discriminative Analysis is done by maximizing Total Scatter Matrix and minimizing Within-Class Scatter Matrix,
finally classification is done by SVM

In implementation I have compared the proposed method results with classical LDA
