import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 


def plot_one_main_three_sub(s, pdf):
    
    # Plot price, and KAMA, then on the bottom make a sub-plot of macd 
    fig, ax = plt.subplots(4,1, figsize=(15,10), sharex=True, gridspec_kw={'height_ratios': [4, 1, 1, 1]})
    ax[1].plot(pdf.macd, label='macd')
    ax[1].plot(pdf.macd_signal, label='signal')
    ax[1].legend(loc = 'upper left')
    ax[0].plot(pdf.Close, label='Close')
    ax[0].plot(pdf.ema_med, label='EMA')
    ax[0].plot(pdf.kama, label='FKAMA')
    ax[0].plot(pdf.kama_slow, label='SKAMA')
    ax[0].legend(loc = 'upper left')
    ax[2].plot(pdf.fast_bb, label='BB')
    ax[2].plot(pdf.fast_kc, label='KC')
    ax[2].legend(loc = 'upper left')
    #ax[3].plot(pdf.fast_kc - pdf.fast_bb, label='KC')
    ax[3].plot(pdf.rsi, label='RSI', c = 'grey')
    ax[3].legend(loc = 'upper left')
    fig.suptitle(s.upper())
    return fig, ax 


