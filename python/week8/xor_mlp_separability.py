from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    files_and_labels = [
        ("evo_result_7_1.0_1.0.npy", 0),
        ("evo_result_7_1.0_20.0.npy", 1),
        ("evo_result_7_20.0_1.0.npy", 2),
        ("evo_result_7_20.0_20.0.npy", 3),
    ]

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for filename, label in files_and_labels:
        arr = np.load(base_dir / filename)
        if arr.ndim != 2:
            raise ValueError(f"{filename} should be 2D, got shape {arr.shape}")

        xs.append(arr)
        ys.append(np.full(arr.shape[0], label, dtype=np.int64))

    x = np.vstack(xs)
    y = np.concatenate(ys)
    return x, y


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    x, y = load_dataset(base_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.4,
        random_state=41,
        stratify=y,
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(32,),
                    activation="relu",
                    solver="adam",
                    max_iter=4000,
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_test, y_pred)

    label_names = ["00", "01", "10", "11"]
    print(f"Dataset shape: X={x.shape}, y={y.shape}")
    print(f"Train size: {x_train.shape[0]}, Test size: {x_test.shape[0]}")
    print(f"Test accuracy: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3]))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_names, digits=4))


if __name__ == "__main__":
    main()
