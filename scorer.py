from sklearn.metrics import r2_score


def score(df,model,feature_list=None):
    """
    df : the df with price_log and features for model
    model : model to be tested
    (optional) feature_list : the columns to be considered for the model
    """
    if feature_list==None:
        feature_list = list(df.columns)
        feature_list.remove("price_log")
    return r2_score(df.price_log,model.predict(df[feature_list]))
