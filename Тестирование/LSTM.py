#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

name_file = f'../Наборы данных/LUNA_candle_1m_50.feather'
best = pd.read_feather(name_file)
# %%
best
# %%
# Классификатор
# В течении следующих 15 минут цена вырастет на 0.2-5.0

