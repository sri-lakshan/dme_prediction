import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/messidor_data.csv')

df.columns = df.columns.str.strip()

train_df, val_df = train_test_split(
    df, 
    test_size=0.2, 
    random_state=42,
    stratify=df[['diagnosis', 'adjudicated_dme']]
)

train_df.to_csv('data/train_data.csv', index=False)
val_df.to_csv('data/val_data.csv', index=False)

print("Training and validation CSV files have been created.")
