"""
data_loader.py
================

이 모듈은 Caltech‑101 이미지 데이터셋을 로드하고 학습용, 검증용, 테스트용으로
분할하는 클래스를 정의한다

train.py와 evaluate.py에서 이 클래스를 사용한다.

사용 방법 예시:

    from ML_CNN.data_loader import DataLoader

    loader = DataLoader(data_dir="/path/to/caltech-101")
    train_ds, val_ds, test_ds, num_classes = loader.load_and_split()

    # train_ds, val_ds, test_ds는 tf.data.Dataset 객체이며
    # num_classes는 Faces_easy 삭제 후 남은 클래스의 수.

클래스 데이터 로더는 다음 기능을 수행한다

* `Faces_easy` 폴더를 제거하여 얼굴 클래스가 모델 성능에 영향을 주지 않도록 한다
* 모든 이미지를 지정한 크기로 리사이즈하고 0–255 범위의 RGB 값을 0–1로
  정규화한다
* 전체 데이터셋을 8:1:1 비율로 섞어서 나누고, 배치 처리 및
  prefetch/caching을 적용해 학습 속도를 향상한다.
"""

from __future__ import annotations

import os
import collections
from pathlib import Path
import shutil
from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt


class DataLoader:
    """
    Caltech‑101 데이터셋을 로드하고 분할하는 클래스.

    Attributes:
        data_dir: Caltech‑101 데이터가 위치한 최상위 폴더 경로.
        img_size: (height, width) 튜플 형태의 리사이즈 크기. 기본값은 (128, 128).
        batch_size: 배치 크기. 기본값은 32.
        seed: 셔플을 위한 난수 시드. 기본값은 123.
    """

    def __init__(
        self,
        data_dir: str,
        img_size: Tuple[int, int] = (128, 128),
        batch_size: int = 32,
        seed: int = 123,
    ) -> None:
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.seed = seed

    def _remove_faces_easy(self) -> None:
        """
        Faces_easy 폴더를 제거.

        Caltech‑101 데이터셋에는 얼굴만 포함된 `Faces_easy` 클래스가 포함되어
        있다. 이 데이터셋은 과제상에서 삭제하라는 언급이 있으므로 삭제한다.
        """
        faces_easy_path = os.path.join(self.data_dir, "Faces_easy")
        if os.path.exists(faces_easy_path):
            shutil.rmtree(faces_easy_path)
            print("Faces_easy 클래스 제거 완료")

    def _visualize_class_distribution(self) -> None:
        """클래스별 이미지 개수 분포를 bar chart로 보여준다.

        이 함수는 시각화를 통해 클래스 불균형을 확인하는 용도로 제공.
        학습에는 직접적으로 사용되지 않음
        """
        class_counts = collections.Counter(
            p.parent.name for p in Path(self.data_dir).rglob("*.jpg")
        )
        plt.figure(figsize=(20, 5))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.xticks(rotation=90)
        plt.title("Class Distribution (after removing Faces_easy)")
        plt.show()

    def load_and_split(
        self, *, visualize: bool = False
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, int]:
        """
        데이터셋을 로드하고 학습/검증/테스트로 분할.

        Args:
            visualize: True로 설정하면 클래스 분포를 시각화. 기본값 False.

        Returns:
            train_ds: 학습용 tf.data.Dataset
            val_ds: 검증용 tf.data.Dataset
            test_ds: 테스트용 tf.data.Dataset
            num_classes: Faces_easy를 제거한 후 남은 클래스 수
        """
        # 얼굴 클래스 삭제
        self._remove_faces_easy()

        # 필요한 경우 시각화
        if visualize:
            self._visualize_class_distribution()

        # raw dataset 로딩 (배치 없이)
        all_ds = image_dataset_from_directory(
            self.data_dir,
            labels="inferred",
            label_mode="categorical",
            seed=self.seed,
            image_size=self.img_size,
            batch_size=None,  # 배치는 나중에
        ).map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))

        total_samples = sum(1 for _ in all_ds)
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        # 데이터셋을 섞고 분할
        all_ds = all_ds.shuffle(total_samples, seed=self.seed)
        train_ds = all_ds.take(train_size)
        val_ds = all_ds.skip(train_size).take(val_size)
        test_ds = all_ds.skip(train_size + val_size)

        # 배치 및 최적화
        train_ds = (
            train_ds.batch(self.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            val_ds.batch(self.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        test_ds = (
            test_ds.batch(self.batch_size)
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )

        # 클래스 수 계산
        num_classes = len(
            [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        )
        return train_ds, val_ds, test_ds, num_classes