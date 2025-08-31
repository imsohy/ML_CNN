"""
evaluate.py
===========

model evaluation 스크립트.

저장된 가중치를 로드하고 테스트 데이터셋에 대해
Confusion Matrix와 F1 점수를 계산. 

학습 시 사용한 모델 구조와
하이퍼파라미터를 동일하게 설정하도록 유의

사용 예:

    python evaluate.py --data-dir /path/to/caltech-101 --weights-path best_model.weights.h5 \
                       --conv-blocks 4 --base-filters 64 --dropout 0.3 \
                       --optimizer Adam --lr 1e-3

이 스크립트는 모델 구조를 다시 빌드한 후 지정한 가중치를 불러옴. 
그 뒤 DataLoader를 통해 테스트 데이터를 불러와 confusion matrix를 그림으로
저장하고 F1 점수를 출력.
"""

from __future__ import annotations

import argparse
from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import optimizers
from data_loader import DataLoader
from cnn_model import CNNModel


def parse_args() -> argparse.Namespace:
    """
    명령줄 인자를 파싱.
    """
    parser = argparse.ArgumentParser(description="Caltech‑101 CNN 모델 평가 스크립트")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Caltech‑101 데이터셋이 위치한 디렉터리 경로",
    )
    parser.add_argument(
        "--weights-path",
        type=str,
        required=True,
        help="저장된 모델 가중치(.h5) 파일 경로",
    )
    # 모델 구조 및 하이퍼파라미터 설정
    parser.add_argument("--conv-blocks", type=int, required=True, help="컨볼루션 블록 수")
    parser.add_argument("--base-filters", type=int, required=True, help="첫 블록 필터 수")
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout 비율 (0.0 ~ 1.0). 학습 시 사용한 값과 동일해야 합니다.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        help="학습 시 사용한 optimizer 이름 (예: Adam, RMSprop)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="학습 시 사용한 learning rate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 데이터 로딩 (test 데이터만 필요하지만 split 편의상 전체 호출)
    loader = DataLoader(data_dir=args.data_dir, img_size=(128, 128), batch_size=32, seed=123)
    _, _, test_ds, num_classes = loader.load_and_split()

    # CNN 모델 초기화. results_dir는 불필요하지만 인자로 전달
    model_manager = CNNModel(input_shape=(128, 128, 3), num_classes=num_classes, results_dir="results")

    # 지정된 파라미터로 모델 빌드 및 컴파일
    model = model_manager.build_model(args.conv_blocks, args.base_filters, args.dropout)
    optimizer = getattr(optimizers, args.optimizer)(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # 가중치 로드
    model.load_weights(args.weights_path)
    print(f"가중치를 {args.weights_path}에서 로드했습니다.")

    # 평가 수행
    cm, f1 = model_manager.evaluate_model(model, test_ds, cm_path="confusion_matrix.png")
    print("Confusion Matrix:\n", cm)
    print(f"F1 Score: {f1:.4f}")


if __name__ == "__main__":
    main()