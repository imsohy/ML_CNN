"""
cnn_model.py
============

이 모듈은 Convolutional Neural Network(CNN) 모델의 생성, 하이퍼파라미터 실험,
학습 및 평가를 담당한다.

주요 기능은 다음과 같다.

* 임의의 레이어를 포함한 데이터 증강 레이어(수평 뒤집기, 회전, 확대/축소,
  대비 변화)를 반환하는 `get_augmentation` 메서드.
* 컨볼루션 블록 수, 초기 필터 수, dropout 비율을 매개변수로 받아 CNN
  모델을 생성하는 `build_model` 메서드.
* 전달된 하이퍼파라미터 그리드에 대해 모든 조합을 학습하고, 각 조합에 대해
  검증 정확도와 F1 점수를 계산하여 최고의 조합을 찾는
  `experiment_hyperparams` 메서드.
* best 파라미터를 사용해 최종 모델을 학습하고 가중치를 저장하는
  `train_final_model` 메서드.
* 테스트 데이터셋에 대한 confusion matrix와 F1 점수를 계산하고
  confusion matrix를 이미지 파일로 저장하는 `evaluate_model` 메서드.

사용 예시: `train.py`와 `evaluate.py` 참조
"""

from __future__ import annotations

import os
import itertools
from typing import Dict, Tuple, Iterable, List, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt


class CNNModel:
    """CNN 모델의 생성과 학습/평가를 담당하는 클래스.

    Attributes:
        input_shape: 이미지 입력의 모양. 예: (128, 128, 3).
        num_classes: 분류할 클래스 수.
        results_dir: 결과 파일을 저장할 폴더. 없으면 생성. 기본값 'results'.

    results_dir는 하이퍼파라미터 튜닝 시 중간 가중치 파일을 저장한다.
    
    """

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        results_dir: str = "results",
    ) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        """
        “입력 크기/클래스 수/결과 저장 경로”를 받아 인스턴스 상태를 초기화한 뒤, 
        결과 저장 폴더를 미리 만들어 둠.
        """

    def get_augmentation(self) -> models.Sequential:
        """
        데이터 증강 레이어를 반환.

        Returns:
            tf.keras.Sequential: 증강 레이어 시퀀스.
        """
        return models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])

    def build_model(
        self, conv_blocks: int, base_filters: int, dropout_rate: float
    ) -> models.Model:
        """
        컨볼루션 신경망 모델을 생성.

        Args:
            conv_blocks: 컨볼루션 블록의 개수.
            base_filters: 첫 블록의 필터 수. 이후 블록에서는 두 배씩 증가.
            dropout_rate: dropout 비율. 0이면 dropout 미적용.  

        Returns:
            tf.keras.Model: 생성된 CNN 모델.
        """
        model = models.Sequential()
        model.add(layers.Input(shape=self.input_shape))


        #데이터 증강은 main에서 받아 미리 실행
        # 컨벌루션 블록 반복
        for i in range(conv_blocks):
            filters = base_filters * (2 ** i)
            model.add(layers.Conv2D(filters, (3, 3), padding="same"))
            model.add(layers.BatchNormalization())
            model.add(layers.ReLU())
            model.add(layers.MaxPooling2D((2, 2)))
            
        # GAP + Dense(256) + Dropout + Softmax
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation="relu"))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
        model.add(layers.Dense(self.num_classes, activation="softmax"))

        # 기본 optimizer는 Adam에 learning_rate=1e-3. 
        # 변경하고 싶을 시, training에서 compile을
        # 호출하는 메서드에서 재설정 가능
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def experiment_hyperparams(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        param_grid: Dict[str, Iterable[Any]],
    ) -> Tuple[Dict[str, Any], float, float]:
        """
        하이퍼파라미터 탐색(튜닝) 수행.

        여러 파라미터 조합에 대해 학습을 수행한 뒤, 검증 정확도와 F1 점수를
        기준으로 가장 좋은 조합을 반환.

        Args:
            train_ds: 학습 데이터셋.
            val_ds: 검증 데이터셋.
            param_grid: 실험할 하이퍼파라미터 그리드. 각 키는 파라미터 이름,
                값은 가능한 값의 리스트.

        Returns:
            best_params: 최상의 파라미터 조합 (dict).
            best_val_acc: 해당 조합의 검증 정확도 (float).
            best_f1: 해당 조합의 F1 점수 (float).
        """
        # 단일 값은 리스트로 감싸기
        grid = {
            k: (v if isinstance(v, (list, tuple)) else [v]) for k, v in param_grid.items()
        }
        keys, values = zip(*grid.items())
        combos = [dict(zip(keys, vals)) for vals in itertools.product(*values)]
        results: List[Tuple[Dict[str, Any], float, float]] = []

        for params in combos:
            # 세션 초기화하여 메모리 절약
            tf.keras.backend.clear_session()

            #하이퍼파라미터를 받아와 모델 빌드
            model = self.build_model(
                conv_blocks=params["conv_blocks"],
                base_filters=params["base_filters"],
                dropout_rate=params["dropout"],
            )
            #옵티마이저 설정
            optimizer = getattr(optimizers, params["optimizer"])(learning_rate=params["lr"])
            model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

            #특정 상태 도달 확인 시 
            #early stopping: tolerance 도달 시 중단
            #reduce learning rate on plateau: validation loss의 감소가 학습률이 너무 커서 최적값 근처에서 
            #진동하고 있다고 확인하고, 러닝 레이트를 절반으로 줄임
            #patience 값은 기준 epoch
            cb: List[callbacks.Callback] = [
                callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=2, verbose=1
                ),
            ]
            history = model.fit(
                train_ds,
                epochs=params["epochs"],
                validation_data=val_ds,
                callbacks=cb,
                verbose=1,
            )

            # validation accuracy의 마지막 3 epoch 평균
            val_acc = float(np.mean(history.history["accuracy"][-3:]))

            # validation F1 점수 계산
            y_true: List[int] = []
            y_pred: List[int] = []
            for x_batch, y_batch in val_ds:
                preds = model.predict(x_batch, verbose=0)
                y_true.extend(np.argmax(y_batch.numpy(), axis=1))
                y_pred.extend(np.argmax(preds, axis=1))
            f1 = float(f1_score(y_true, y_pred, average="macro"))

            results.append((params, val_acc, f1))

            # 조합 이름을 만들어 임시 가중치를 저장
            name = f"cb{params['conv_blocks']}_bf{params['base_filters']}_dr{params['dropout']}_opt{params['optimizer']}"
            weight_path = os.path.join(self.results_dir, f"{name}.weights.h5")
            model.save_weights(weight_path)
            print(
                f"실험 완료 {name}: 검증 정확도 {val_acc:.4f}, F1 score {f1:.4f}"
            )

        # F1 score 기준으로 가장 좋은 결과 선택
        best_params, best_val_acc, best_f1 = max(results, key=lambda item: item[2])
        print(
            f"최적 파라미터 {best_params}, 검증 정확도 {best_val_acc:.4f}, F1 score {best_f1:.4f}"
        )
        return best_params, best_val_acc, best_f1

    def train_final_model(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        best_params: Dict[str, Any],
        *,
        weights_path: str = "best_model.weights.h5",
        verbose: int = 0,
    ) -> models.Model:
        """
        최적 파라미터로 모델을 학습하여 가중치를 저장.

        Args:
            train_ds: 학습 데이터셋.
            val_ds: 검증 데이터셋.
            best_params: 하이퍼파라미터 딕셔너리.
            weights_path: 학습 완료 후 저장할 가중치 파일 경로. 기본값 'best_model.weights.h5'.
            verbose: keras fit 함수의 verbose 설정. 기본값 0.

        Returns:
            model: 학습이 완료된 tf.keras.Model 객체.
        """
        model = self.build_model(
            conv_blocks=best_params["conv_blocks"],
            base_filters=best_params["base_filters"],
            dropout_rate=best_params["dropout"],
        )
        optimizer = getattr(optimizers, best_params["optimizer"])(learning_rate=best_params["lr"])
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
        cb: List[callbacks.Callback] = [
            callbacks.EarlyStopping(
                monitor="val_loss", patience=6, restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, verbose=1
            ),
        ]
        model.fit(train_ds, epochs=best_params["epochs"], validation_data=val_ds, callbacks=cb, verbose=verbose)
        model.save_weights(weights_path)
        return model

    def evaluate_model(
        self, model: models.Model, test_ds: tf.data.Dataset, *, cm_path: str = "confusion_matrix.png"
    ) -> Tuple[np.ndarray, float]:
        """
        테스트 데이터셋에 대해 모델을 평가.

        Args:
            model: 학습이 완료된 모델. `build_model`로 생성 후 `load_weights`하거나 `train_final_model`
                결과를 사용할 수 있음.
            test_ds: 테스트 데이터셋.
            cm_path: 저장할 confusion matrix 이미지 파일 이름. 기본값 'confusion_matrix.png'.

        Returns:
            cm: confusion matrix (numpy.ndarray).
            f1: macro averaged F1 score (float).
        """
        y_true: List[int] = []
        y_pred: List[int] = []
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(preds, axis=1))
        cm = confusion_matrix(y_true, y_pred)
        f1 = float(f1_score(y_true, y_pred, average="macro"))

        # confusion matrix 저장
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(cm_path)
        plt.close()

        return cm, f1