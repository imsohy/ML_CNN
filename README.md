# ML_CNN
Kaggle Caltech-101 dataset


프로그램 파이프라인 개요:
    1. 다운받은 데이터에서 Faces_easy 폴더는 제외한다.
    2. 이미지 크기를 128x128로 리사이즈하고, 픽셀 RGB값을 0~255 -> 0~1 범위로 정규화한다.
    3. 데이터셋을 학습(train), 검증(validation), 테스트(test) 용으로 8:1:1 비율로 분할한다.
    4. train 데이터는 좌우 반전, 회전, 확대 축소, 대비 변화로 Data Augumentation을 수행한다.
    5. 다양한 하이퍼파라미터(conv_blocks, base_filters, dropout, optimizer, learning rate, epochs) 조합에 대해
       학습을 수행한다. 각 학습에 일정 iteration동안 cost function 값이 더 작아지지 않으면 정지한다. (keras callbacks.EarlyStopping)
       또한 
       검증 정확도를 기준으로 최적의 모델을 찾는다.
       5.1. 모델의 구조에 큰 영향을 주는 conv_blocks, base_filters의 조합만 바꿔가며 수행. (나머지는 고정)
       5.2. dropout, optimizer, learning rate, epochs의 조합을 바꿔가며 수행.
    6. 최적의 하이퍼파라미터로 전체 train / validation 데이터로 다시 학습하여 best_model.h5로 저장한다.
    7. test 데이터로 예측을 수행하고 confusion_matrix.png과 F1 점수를 출력한다.
