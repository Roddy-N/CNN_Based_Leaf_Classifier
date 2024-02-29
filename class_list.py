import pandas as pd

def class_list():
    df = pd.read_csv('/home/wolf/Desktop/TK tutorial/train.csv', index_col=False)
    dftest = pd.read_csv('/home/wolf/Desktop/TK tutorial/test.csv', index_col=False)
    
    classes = df['species'].unique().tolist()

    return classes

