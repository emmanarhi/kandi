import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller


def plotting(yx, names):
    # Plot
    fig, axs = plt.subplots(len(yx), figsize=(10, 10))

    for i in range(len(yx)):
        axs[i].plot(yx[i])
        axs[i].set_ylabel(names[i])
        axs[i].set_xlabel("time")


def numpify(names, df):
    y = []
    for i in range(len(names)):
        y.append(df[names[i]].to_numpy())

    return y


def adfuller_tests(yx, names):
    i = 0
    for y in yx:
        result = adfuller(y)
        print("p-test result for {} : {:.2e}".format(names[i], result[1]))
        i += 1


def main():
    df = pd.read_csv("dataJan-Aug2023.csv", sep=";", decimal=',')
    columns = ['temp before VE', 'flow rate', 'valve feedback', 'return temp', 'temp after VE', 'power', 'flow rate',
               'supply temp', 'return temp']
    df.columns = ['time'] + columns

    df['time'] = pd.to_datetime(df['time'], dayfirst=True)
    df.index = df['time']
    del df['time']


    df = df.replace(',', '.', regex=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    names1 = columns[0:5]
    names2 = columns[5:9]

    y1 = numpify(names1, df.iloc[:, 0:5])
    y2 = numpify(names2, df.iloc[:, 5:9])

    # Augmented dickey fuller test
    # H0: The time series is non-stationary
    # H1: The time series is stationary
    # Significance level = 0.05

    adfuller_tests(y1, names1)
    adfuller_tests(y2, names2)

    # p << 0.05 for all -> Reject null -> Series are stationary?

    # Plots
    plotting(y2, names2)
    plotting(y1, names1)
    plt.show()

    # Normal Pearson correlations, finding linear trends

    corr = np.corrcoef(y1, y2)
    print(corr)
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlations")
    plt.show()

    # Granger-causality tests even though they are considered dumb by The Granger himself for applications outside
    # eCoNoMeTrIcSsS.... but i think they give a hint i think do i need this idk i think?

    

if __name__ == "__main__":
    main()
