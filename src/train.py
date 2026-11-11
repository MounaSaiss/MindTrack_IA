
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import prepare_data


def train_models():

    # Prepare training data
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    }

    # Create pipelines
    pipelines = {
        name: Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
        for name, model in models.items()
    }

    # Logistic Regression optimization
    param_grid_lr = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__max_iter": [100, 200, 500, 1000, 2000, 3000]
    }

    grid_lr = GridSearchCV(
        estimator=pipelines["Logistic Regression"],
        param_grid=param_grid_lr,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_lr.fit(X_train, y_train)

    # Random Forest optimization
    param_grid_rf = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [None, 10, 20, 30],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4]
    }

    grid_rf = GridSearchCV(
        estimator=pipelines["Random Forest"],
        param_grid=param_grid_rf,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1
    )

    grid_rf.fit(X_train, y_train)

    # Select the final model
    best_model = grid_lr.best_estimator_

    # Save the complete pipeline
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "student_mental_health_model.pkl"

    joblib.dump(best_model, model_path)

    print("\nTrain Done ")


if __name__ == "__main__":
    train_models()
