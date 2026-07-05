import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def run_preprocessing(input_path, output_dir):
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Preprocessing data...")
    # Drop irrelevant columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Encode categorical features
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    df['Embarked'] = le.fit_transform(df['Embarked'])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'titanic_clean.csv')
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    input_file = '../titanic_raw.csv'
    output_folder = 'titanic_preprocessing'
    
    # adjust path if run from root
    if not os.path.exists(input_file):
        input_file = 'titanic_raw.csv'
        output_folder = 'preprocessing/titanic_preprocessing'
        
    run_preprocessing(input_file, output_folder)
