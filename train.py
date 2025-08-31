"""
train.py
========

모델 학습 스크립트.

Caltech‑101 데이터셋을 로드하고, 선택적으로 하이퍼파라미터
탐색을 수행한 뒤 최적 모델을 학습하여 가중치 파일로 저장. 데이터
로딩과 모델 생성은 "data_loader.py"와 "cnn_model.py"에서 정의된 클래스를 사용.

사용 예:

    python train.py --data-dir /path/to/caltech-101 --augment --search

명령줄 인자 설명:

* --data-dir: Caltech‑101 데이터셋 폴더 경로. 필수.
* --augment: 학습 데이터에 데이터 증강을 적용할지 여부. 기본값은 적용.
* --search: 전체 하이퍼파라미터 그리드 탐색을 수행할지 여부. 기본값은
  False로, 사전에 결정된 best 파라미터를 사용.
* --results-dir: 임시 가중치를 저장할 디렉터리. 기본값 'results'.

이 스크립트는 학습 완료 후 "best_model.weights.h5" 파일을 현재 작업
디렉터리에 저장합. 

모델 아키텍처는 "cnn_model.py"에서 정의되며,
별도로 저장하지는 않음. 

추후 평가 시, 동일한 파라미터로 모델을
구성하고 "load_weights"를 호출하면 됨.
"""

from __future__ import annotations

import argparse
from typing import Dict

import tensorflow as tf

from data_loader import DataLoader
from cnn_model import CNNModel


# 하이퍼파라미터 그리드 기본값
PARAM_GRID: Dict[str, list] = {
    "conv_blocks": [3, 4],
    "base_filters": [32, 64],
    "dropout": [0.0, 0.3, 0.5],
    "optimizer": ["Adam", "RMSprop"],
    "lr": [1e-3, 1e-4],
    "epochs": [30, 50],
}

# 사전 탐색된 최적 하이퍼파라미터
DEFAULT_BEST_PARAMS: Dict[str, object] = {
    "conv_blocks": 4,
    "base_filters": 64,
    "dropout": 0.3,
    "optimizer": "Adam",
    "lr": 1e-3,
    "epochs": 30,
}


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    parser = argparse.ArgumentParser(description="Caltech‑101 CNN 모델 학습 스크립트")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Caltech‑101 데이터셋이 위치한 디렉터리 경로",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        default=False,
        help="학습 데이터에 데이터 증강을 적용합니다. 기본값은 비활성화.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        default=False,
        help="하이퍼파라미터 탐색을 수행합니다. 기본값은 사전에 결정된 파라미터를 사용.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="임시 가중치를 저장할 디렉터리",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 데이터 로딩
    loader = DataLoader(data_dir=args.data_dir, img_size=(128, 128), batch_size=32, seed=123)
    train_ds, val_ds, test_ds, num_classes = loader.load_and_split()

    # 모델 초기화
    model_manager = CNNModel(input_shape=(128, 128, 3), num_classes=num_classes, results_dir=args.results_dir)

    # 데이터 증강 적용 여부
    if args.augment:
        aug = model_manager.get_augmentation()
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
        ).cache().prefetch(tf.data.AUTOTUNE)

    # 하이퍼파라미터 탐색
    if args.search:
        print("하이퍼파라미터 탐색을 시작합니다. 전체 그리드를 사용하면 시간이 오래 걸릴 수 있습니다.")
        best_params, _, _ = model_manager.experiment_hyperparams(train_ds, val_ds, PARAM_GRID)
    else:
        print("사전 정의된 하이퍼파라미터를 사용하여 학습을 진행합니다.")
        best_params = DEFAULT_BEST_PARAMS

    print(f"사용할 최적 파라미터: {best_params}")

    # 최종 모델 학습
    model = model_manager.train_final_model(train_ds, val_ds, best_params, weights_path="best_model.weights.h5", verbose=1)
    print("모델 학습이 완료되었습니다. weights는 'best_model.weights.h5'에 저장됩니다.")


if __name__ == "__main__":
    main()