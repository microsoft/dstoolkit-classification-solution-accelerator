import pandas as pd
import numpy as np

def preprocessing(data, keep_columns, target_column, target_values):
    """
    Create Week_number from WeekStarting
    Drop two unnecessary columns: WeekStarting, Revenue
    """
    df = data[keep_columns].fillna('other')

    # First, we need to address any imbalance in the data by samping down the majority class. 
    #   NOTE- this method needs at least around 1k rows in the minority class
    #https://machinelearningmastery.com/what-is-imbalanced-classification/
    r1 = pd.DataFrame(np.random.uniform(0,1,len(df)))
    df['rand'] = r1  
    imbalance = (len(df.loc[df[target_column] == target_values[0]])/len(df))
    
    if imbalance > .6:
        x = (1 - imbalance)/(imbalance)
        df['Flagged'] = 0
        df.loc[(df[target_column]== target_values[0]) & (df['rand']>= x), 'Flagged'] = 1 
        df = df.loc[df['Flagged'] == 0]
        df = df.loc[df['Flagged'] == 0].drop(columns = ['Flagged', 'rand']) 
    elif imbalance < .4:
        x = imbalance/(1 - imbalance)
        df['Flagged'] = 0
        df.loc[(df[target_column]== target_values[1]) & (df['rand']>= x), 'Flagged'] = 1 
        df = df.loc[df['Flagged'] == 0].drop(columns = ['Flagged', 'rand']) 
    
    return df


def predict(model, data):
    # Perform predictions and add three new columns to the Pandas dataframe
    data['Prediction'] = model.predict(data)
    data[['Probability4True', 'Probability4False']] = pd.DataFrame(model.predict_proba(data))

    return data
