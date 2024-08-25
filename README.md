Implementation of paper: Deep residual network with regularised fisher framework for detection of Melanoma

The Paper proposes to extract features from pretrained on Imagenet, finetuned on ISBI dataset ResNet50 model, 
further Discriminative Analysis is done by maximizing Total Scatter Matrix and minimizing Within-Class Scatter Matrix,
finally classification is done by SVM

In implementation I have compared the proposed method results with classical LDA
