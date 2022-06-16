"""
Input:  finance TimeSeries
Output: array classification
"""
#%%
import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit

def show_pie(data):
    plt.figure(figsize=(8, 8))
    plt.pie(np.bincount(data), autopct='%1.1f%%')
    plt.show()

def show_classification(Close, cls):
    import plotly.graph_objects as go
    from plotly.offline import iplot
    import plotly.express as px
    from plotly.subplots import make_subplots
    Close = Close[:2000]#int(len(Close)/10)]
    color = {   '0':'Blue',
                '1':'Green',
                '2':'Red', 
                '3':'Black',
                '4':'Purple',
                '5': 'Yelow'
    }   
    p = np.arange(0, len(cls), 1)
    price=go.Scatter(x=p,y=Close, name="Close", mode='lines', marker=dict(color=color['0'], size=2))
    data=[price]
    buy_index = [i for i,d in enumerate(cls) if d == 1]
    cls_buy = [d for i,d in enumerate(Close) if i in buy_index]
    cls_scat=go.Scatter(x=buy_index,y=cls_buy, name="buy", mode='markers', marker=dict(color=color['1'], size=4))
    data.append(cls_scat)
    sell_index = [i for i,d in enumerate(cls) if d == 2]
    cls_sell = [d for i,d in enumerate(Close) if i in sell_index]
    cls_scat=go.Scatter(x=sell_index,y=cls_sell, name="sell", mode='markers', marker=dict(color=color['2'], size=4))
    data.append(cls_scat)
    fig = make_subplots(rows=1, cols=1, subplot_titles=("Классификация"))
    for x in data:
        fig.add_trace(x,row=1, col=1)
    fig.update_layout(  autosize=False,
                        width=800,
                        height=400,)
    fig.show()

def cls_down_up(df):
    """ 
    0 - Если следующий бар падает
    1 - Если Следующий бар растёт
    """
    print('Начало классификации: Подъём падение')
    Close = df['Close'].to_list()
    
    @njit()
    def cls_(Close):
        cls = [2] * (len(Close)-1)
        for i in range(len(Close)-1):
            if Close[i+1] > Close[i]: cls[i] = 1
        return cls
    cls = cls_(Close=Close)
    """ Удалять последние бары которые нельзя классифицировать """
    show_pie(cls)
    df = df[:df.shape[0]-1]
    return df, cls

def per_Open_Close(df):
    k = 0.02
    df['OC'] = (df['Close']/df['Open'] - 1)*100
    return df['OC'].to_list()

def cls_persent_up_down(df, pr, bar):
    """
    pr - проценты;
    bar - за сколько баров

    2 - если за столько bar цена изменяется на pr.% down
    1 - если за столько bar цена изменяется на pr.% up
    0 - всё остальное

    (df, 0.5, 5)
    """
    Close = df['Close'].to_list()
    OC = per_Open_Close(df)

    # @njit()
    def cls_(Close, OC, pr, bar):
        cls = [0] * (len(Close)-bar)
        for i in range(len(OC)-bar):
            oc_ = OC[i:i+bar]
            # sum_ = sum(oc_)
            if Close[i+bar] > Close[i]: 
                if sum([i for i in oc_ if i < 0]) > (-1)*pr/3:
                    cls[i] = 1
            if Close[i+bar] < Close[i]: 
                if sum([i for i in oc_ if i > 0]) < pr/3:
                    cls[i] = 2
        return cls
    cls = cls_(Close=Close, OC=OC, pr=pr, bar=bar)
    """ Удалять последние бары которые нельзя классифицировать """
    show_pie(cls)
    show_classification(Close, cls)
    df = df[:df.shape[0]-bar]
    return df, cls

#%%
if __name__ == "__main__":
    import dataset
    df = dataset.dowload('LUNA', 'test')
    %time cls_persent_up_down(df, 0.5, 3)