import pandas as pd
from pathlib import Path

def load_data():
    project_root=Path(__file__).resolve().parent.parent
    
    data_path = (project_root   / "DatasetMental_Health" / "student_mental_health.csv")
    
    df=pd.read_csv(data_path)
    
    return df

if __name__ == "__main__":
    df = load_data()
    
    print("Data Loaded ")
    print(df.head())
    print("Shape of the dataset:", df.shape)