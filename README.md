# CNN

## 개요

본 프로젝트는 Kaggle Caltech-101 데이터셋을 사용하여, 다양한 사물 이미지를 분류하는 Convolutional Neural Network(CNN) 모델을 구현하고 그 성능을 평가했다. 

Caltech-101은 총 101개의 class와 하나의 background 클래스 (Background_Google)을 포함하며, 각 클래스는 40~800개 사이 이미지로 구성되어있다. 모든 이미지는 크기와 배경이 다양하며 컬러 RGB 이미지로 되어있다.

# How to Run

<img width="290" height="381" alt="image" src="https://github.com/user-attachments/assets/392a043e-0bba-488d-b7bd-a1f85bfdd739" />

위의 이미지에서 가중치 (.h5) 파일은 train 결과 후에 생성되는 것이다.
https://colab.research.google.com/drive/1EulSqo02vUvaPxI_mxN4PmKgGl9uC-Lt
google colab에서 자세한 실행 과정을 확인하고 실험해볼 수 있다.

## dataset 

using Kaggle Caltech-101 dataset

https://www.kaggle.com/datasets/imbikramsaha/caltech-101?resource=download

## Result

<img width="1191" height="501" alt="image" src="https://github.com/user-attachments/assets/74eb6600-9eeb-484e-9916-656bf1a5823d" />




<img width="1000" height="800" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d9dc0aa4-b56b-4d78-a5d1-3ab54d4b518b" />

F1 Score: 0.1382
