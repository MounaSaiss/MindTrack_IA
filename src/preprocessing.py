
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,OrdinalEncoder,StandardScaler)
from data_preparation import load_data

def prepare_data():
    df=load_data()
    
    #Separation des Feautures 
    X = df.drop(columns=["Mental_Health_Risk", "Student_ID"])
    y = df["Mental_Health_Risk"]
    
    #Split data 
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Defenir les  columns
    num_cols = [
        "Age", "Year_of_Study", "Sleep_Hours",
        "Study_Hours_Per_Week", "GPA",
        "Social_Media_Hours", "Screen_Time_Hours",
        "Financial_Stress", "Family_Support",
        "Attendance_Rate"
    ]

    nominal_cols = [
        "Gender",
        "Part_Time_Job",
        "Previous_Mental_Health_History"
    ]

    ordinal_cols = [
        "Physical_Activity",
        "Diet_Quality"
    ]

    
    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    
    
    # Nominal preprocessing
    nominal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    
    # Ordinal preprocessing
    ordinal_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[
                ["Low", "Moderate", "High"],
                ["Poor", "Average", "Good"]
            ]
        ))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("nominal", nominal_pipeline, nominal_cols),
            ("ordinal", ordinal_pipeline, ordinal_cols)
        ],
        remainder="passthrough"
    )

    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":

    X_train, X_test, y_train, y_test, preprocessor = prepare_data()

    print("Data preprocessing Done ")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("valeurs manques dans train:", X_train.isnull().sum().sum())
    print("Valeurs manques dans test :", X_test.isnull().sum().sum())