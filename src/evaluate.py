
from pathlib import Path
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from preprocessing import prepare_data


def evaluate_model():

    # Load test data
    X_train, X_test, y_train, y_test, _ = prepare_data()

    # Load trained model
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "student_mental_health_model.pkl"

    model = joblib.load(model_path)

    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Calculate evaluation metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="macro"),
        "Recall": recall_score(y_test, y_pred, average="macro"),
        "F1-Score": f1_score(y_test, y_pred, average="macro"),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_proba,
            multi_class="ovr",
            average="macro"
        )
    }


    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return metrics


if __name__ == "__main__":
    evaluate_model()
