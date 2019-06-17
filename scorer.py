from sklearn.metrics import r2_score
import pandas as pd
import numpy as np


def score(df,model,feature_list=None,y=None):
    """
    df : the df with price_log and features for model
    model : model to be tested
    (optional) feature_list : the columns to be considered for the model
    """
    if y is not None:
        return r2_score(y,model.predict(df))
    if feature_list==None:
        feature_list = list(df.columns)
        feature_list.remove("price_log")
    return r2_score(df.price_log,model.predict(df[feature_list]))
def add_binary_basement(df):
    df['basement'] = [1 if df.iloc[n]['sqft_basement'] !=0 else 0 for n in range(0,len(df))]
    return df
def make_season(df):
    df['date'] = pd.to_datetime(df['date'])


    # Build a season_column
    # assigning season to month of year
    seasons = ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Winter']
    

    season_col = df['date'].map(lambda x: seasons[x.month-1] )
    
    season_dummies = pd.get_dummies(season_col, prefix="season_")
    for col in season_dummies.columns:
        df[col] = season_dummies[col]
    return df


from sklearn.model_selection import KFold

    
def k_fold_test(model,x,y,fold_count):
    kf = KFold(n_splits=fold_count, random_state=None, shuffle=False)
    scores = []
    for train_index, test_index in kf.split(x):
        model.fit(x[train_index],y[train_index])

        scores.append(score(x[test_index],model,y=y[test_index]))
    return np.mean(scores)
