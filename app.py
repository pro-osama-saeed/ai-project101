import pandas as pd


def train_model(df):
    # Old code
    # X = df[['Size', 'Rooms', 'Age']]
    # Updated code
    X = df[['size', 'rooms', 'age']]
    # Old code
    # y = df['Price']
    # Updated code
    y = df['price']
    
    # Rest of your training code goes here
    

if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    train_model(data)