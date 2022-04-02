# Kaggle Histopathologic Cancer Detection

Tackling the Kaggle [Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/overview) Challenge to evaluate different machine-learning algorithms for identifying metastatic cancer in small image patches taken from larger digital pathology scans.

## Goal

For this project, in order to understand how far traditional Computer-Vision techniques have evolved with the advent of Deep Learning, I start from the very basic of algorithms and iteratively improve each model's performance one small step at a time. Not all steps are guaranteed to improve performance, but it's necessary to try them to build a working intuition of what might work.

I start off with hand-engineered CV features (Color-Space Transforms, LBP, Gabor, Scharr, Laplacian, Harris, etc.) that work well with Shallow-ML models, and compare their performance against the automatic feature-extraction of large DL models.

## Results

Validation accuracy of the baseline model the started out at ***53.2%***. The best Shallow-ML model topped out at ***87.2%*** using 60 hand-engineered features. The best CNN model topped out at ***97.6%***.

## Journey

---

| Step | Notebook | Description |
| --- | --- | --- |
| 1 | [Data_Exploration](Data_Exploration.ipynb) | Exploratory Data Analysis |
| 2 | [Data_HDF5](Data_HDF5.ipynb) | Generate Grayscale+[HED](https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2hed) [HDF5](https://www.hdfgroup.org/solutions/hdf5/) dataset volume |
| 3 | [Data_1D](Data_1D.ipynb) | Generate Naïve-1D flattened .npz from HDF5 for Shallow-ML |
| 4 | [LogReg](LogReg.ipynb) | Baseline Naïve-1D with Logistic Regression |
| 5 | [Create_LBP_Feat](Create_LBP_Feat.ipynb) | Generate [LBP](https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=local_binary_pattern#skimage.feature.local_binary_pattern) features and Evaluate on [GBT](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) classifier |
| 6 | [LBP_Euclidean_vs_KLD](LBP_Euclidean_vs_KLDivergence.ipynb) | LBP histogram Dissimilarity metrics: Euclidean vs KL-Divergence |
| 7 | [LogReg](LogReg.ipynb) | Baseline LBP features with Logistic Regression |
| 8 | [Find_Landmarks](Find_Landmarks.ipynb) | Develop/Test algorithmn for finding set of 'Landmarks' |
| 9 | [Generate_Landmarks](Generate_Landmarks.ipynb) | Generate Landmarks on Histopathology dataset LBP features |
| 10 | [Create_LDist_Feat](Create_LDist_Feat.ipynb) | Generate Distance-to-Landmarks (identified above) features |
| 11 | [LogReg](LogReg.ipynb) | Baseline Landmark features with Logistic Regression |
| 12 | [PCA](PCA.ipynb) | Evaluate effect of PCA transformation on i. LBP and ii. Landmark features |
| 13 | [SVM](SVM.ipynb) | Evaluate SVM model with i. LBP and ii. Landmark features|
| 14 | [Create_5LBP_Feat](Create_5LBP_Feat.ipynb) | Generate 5-cell overlapping LBPs: 64x64px centered and 32x32px on four corners |
| 15 | [GBT](GBT.ipynb) | Evaluate GBT model with 'Double-LBP' (scaling-pyramid: full-size 96x96px, half-size 48x48px) features |
| 16 | [Create_COPOD_Feat](Create_COPOD_Feat.ipynb) | Classification using [COPOD](https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.copod.COPOD) scores on LBP features |
| 17 | [Create_2x2LBP_Feat](Create_2x2LBP_Feat.ipynb) | Add 2nd set of [Rotation-Invariant](https://ieeexplore.ieee.org/document/1017623) LBP texture features |
| 18 | [Create_Gabor_Feat](Create_Gabor_Feat.ipynb) | Add Gabor Filters (16x 2-D kernels) features |
| 19 | [Create_Gabor_Scharr_Feat](Create_Gabor_Scharr_Feat.ipynb) | Add Gabor+Scharr Gradient Filter features |
| 20 | [Create_Laplacian_Feat](Create_Laplacian_Feat.ipynb) | Add Laplacian Edge-Detection Filter features |
| 21 | [Create_Harris_Feat](Create_Harris_Feat.ipynb) | Add Harris Corner-Detection Filter features |
| 22 | [GBT](GBT.ipynb) | Re-evaluate GBT model on aggregation of best Shallow-ML features |
| 23 | [TPOT.ipynb](TPOT.ipynb) | Evaluate [TPOT](http://epistasislab.github.io/tpot/) Auto-ML on Shallow-ML features (Last of Shallow Models) |
| 24 | [NN](NN.ipynb) | Evaluate Neural Network with Shallow-ML (LBP, Gabor, Scharr) features|
| 25 | [CNN_ModelA](CNN_ModelA.ipynb) | Sequential CNN with Increasing # Conv2D filters|
| 26 | [CNN_ModelB](CNN_ModelB.ipynb) | Sequential CNN with Decreasing # Conv2D filters|
| 27 | [CNN_ModelA-BD](CNN_ModelA-BD.ipynb) | CNN_ModelA on full 200k Train set |
| 28 | [CNN_ModelD1-BD-AUG-N](CNN_ModelD1-BD-AUG-N.ipynb) | Added Augmentations, Gaussian Noise, more Dropout |
| 29 | [CNN_ModelF](CNN_ModelF.ipynb) | Change last Conv2D from AvgPooling2D to Conv2D, Reduce Learning-Rate |
