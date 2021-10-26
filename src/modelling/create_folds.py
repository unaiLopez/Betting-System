import pandas as pd

from typing import List, Tuple

def create_folds(df: pd.DataFrame) -> List[Tuple]:
    folds = list()
    df = df.reset_index()
    for season in df.Season.unique():
        print(season)
        temp_df = df[df.Season == season]

        test_index = temp_df.sample(frac=0.5, random_state=42).index.values
        train_index = df.index.values[~test_index]
        
        folds.append((train_index, test_index))
    
    return folds