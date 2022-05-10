import pandas as pd
import numpy as np
import xlrd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from scipy.stats import norm
from scipy import stats

frame = pd.read_excel("e:\\result_fj.xls", "Miss")

z_score = frame.copy()
print(z_score.columns)

index = ['AggressionMax',
         'AggressionMin', 'AggressionM', 'AggressionC',
         'StressMax', 'StressMin', 'StressM', 'StressC',
         'TensionMax', 'TensionMin', 'TensionM', 'TensionC',
         'SuspectMax', 'SuspectMin', 'SuspectM', 'SuspectC',
         'BalanceMax', 'BalanceMin', 'BalanceM', 'BalanceC',
         'CharmMax', 'CharmMin', 'CharmM', 'CharmC',
         'EnergyMax', 'EnergyMin', 'EnergyM', 'EnergyC',
         'RegulationMax', 'RegulationMin', 'RegulationM', 'RegulationC',
         'InhibitionMax', 'InhibitionMin', 'InhibitionM', 'InhibitionC',
         'NeuroticismMax', 'NeuroticismMin', 'NeuroticismM', 'NeuroticismC',
         'DepressionMax', 'DepressionMin', 'DepressionM', 'DepressionC',
         'HappinessMax', 'HappinessMin', 'HappinessM', 'HappinessC',
         'Extraversion',
         'Stability',
         'Vi']


def transform(x):
    y = x.rank(method='min')
    eps = 1e-4
    for i, v in enumerate(y):
        p = (v - 1) / (len(y) - 1)
        p = max(p, eps)  # 分位数为0在正态曲线面积表中取-inf
        p = min(p, 1 - eps)  # 同理会取到inf
        y[i] = norm.ppf(p)
    mean, std = y.mean(), y.std()
    y = y.map(lambda x: (x - mean) / std)
    return y


def show_data(x):
    sns.distplot(x, fit=norm)
    # 均值和方差
    (mean, std) = norm.fit(x)
    skew, kurt = x.skew(), x.kurt()
    #     print('\n mean = {:.2f} and std = {:.2f skew = {:.2f} and kurt = {:.2f}}\n'.format(mu, sigma,x.skew(),x.kurt))
    plt.legend(['(mean=$ {:.2f}      std=$ {:.2f} \n skew=$ {:.2f}      kurt=$ {:.2f} )'.format(mean, std, skew, kurt)],
               loc='best')
    plt.ylabel('Frequency')

    fig = plt.figure()
    res = stats.probplot(x, plot=plt)
    plt.show()


for i in index:
    z_score[i] = transform(frame[i])
    show_data(z_score[i])
z_score.to_csv('e:\\result_z_score.csv', encoding="utf_8_sig")
