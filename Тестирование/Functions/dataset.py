#%%
import pandas as pd

data = {
    'train':        '100',
    'test':         '50',
    'valid':        '25'
}

def dowload(Ticket, type):

    import os
    print(os.path.abspath(os.getcwd()))
    try:
        name_file = f'Functions/Datasets/{Ticket}_candle_1m_{data[type]}.feather'
        df = pd.read_feather(name_file)
    except:
        try:
            name_file = f'Datasets/{Ticket}_candle_1m_{data[type]}.feather'
            df = pd.read_feather(name_file)
        except:
            pass

    return df

if __name__ == "__main__":
    print(dowload('LUNA', 'test'))
# %%
