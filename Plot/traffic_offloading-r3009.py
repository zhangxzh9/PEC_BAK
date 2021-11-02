# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/22'

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoLocator, ScalarFormatter
    import sys
    import os 
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes  
    plt.figure(figsize=(20, 13))
    #plt.title('hitrate (city level)',fontsize=40) 
    #optimal      = list(np.array([0.35486,0.84040171,0.91643094,1         ])*100)
    #random       = list(np.array([0.09059, 0.0934 , 0.08723, 0.08825, 0.08818, 0.#08638,  0.08448, 0.08411, 0.08362, 0.08257, 0.08212, 0.08162, 0.08092,
    #    0.08119, 0.08095, 0.08079, 0.08082, 0.08106, 0.08078, 0.08094,
    #    0.08124, 0.08088, 0.08085, 0.0807 , 0.08114, 0.08093, 0.08072,
    #    0.08097, 0.08117, 0.08118])*100)
    efficiency   = list(np.array([0.1488 , 0.22765, 0.2487 , 0.26798, 0.28402, 0.29712, 0.30571,
        0.31949, 0.32678, 0.33408, 0.34065, 0.3471 , 0.35233, 0.36067,
        0.36915, 0.37278, 0.37991, 0.38554, 0.39146, 0.39576, 0.40087,
        0.40356, 0.40562, 0.4081 , 0.41166, 0.41328, 0.41453, 0.41678,
        0.42026, 0.42562, 0.42611])*100)
    privacy      = list(np.array([0.3419 , 0.35205, 0.33187, 0.3227 , 0.3134 , 0.30577, 0.29693,
        0.29249, 0.28614, 0.28163, 0.28056, 0.27695, 0.27263, 0.27114,
        0.26934, 0.26624, 0.26575, 0.26619, 0.26625, 0.26626, 0.26835,
        0.26712, 0.26627, 0.26532, 0.26491, 0.26375, 0.26256, 0.2625 ,
        0.26173, 0.26178, 0.26158])*100)
    Deepcache    = list(np.array([0.1027 , 0.2787 , 0.3097 , 0.31968, 0.3302 , 0.33547, 0.3343 ,
        0.3426 , 0.34427, 0.34667, 0.34958, 0.35123, 0.34905, 0.35439,
        0.35987, 0.35841, 0.36256, 0.36584, 0.36794, 0.36909, 0.37162,
        0.37043, 0.36956, 0.36882, 0.36965, 0.36822, 0.36675, 0.36717,
        0.36978, 0.37353, 0.37384,])*100)
    POC = list(np.array([ 0.0804 , 0.23985, 0.28513, 0.30205, 0.31694, 0.32523, 0.32623,
        0.33614, 0.33918, 0.34247, 0.34616, 0.34844, 0.34689, 0.35267,
        0.35852, 0.35752, 0.36202, 0.36571, 0.36803, 0.36946, 0.37216,
        0.37125, 0.3706 , 0.37005, 0.37103, 0.36971, 0.36836, 0.36886,
        0.37156, 0.37541, 0.37574,])*100)
    noneCache    = list(np.array([0.385  , 0.42745, 0.41123, 0.39848, 0.39628, 0.39353, 0.38591,
        0.38965, 0.38726, 0.38685, 0.38734, 0.38693, 0.38312, 0.38718,
        0.39145, 0.38885, 0.39204, 0.39449, 0.39623, 0.39668, 0.39839,
        0.39664, 0.39534, 0.39411, 0.39457, 0.39255, 0.39052, 0.39051,
        0.39281, 0.39617, 0.39647])*100)
    balance = list(np.array([0.3749 , 0.4447 , 0.44463, 0.44362, 0.45496, 0.45798, 0.45676,
        0.4641 , 0.46584, 0.46937, 0.47235, 0.4726 , 0.47243, 0.47597,
        0.47961, 0.4792 , 0.48278, 0.48523, 0.48704, 0.48739, 0.48757,
        0.48566, 0.48325, 0.4814 , 0.48055, 0.47831, 0.47608, 0.4751 ,
        0.47658, 0.47855, 0.47868])*100)

    x = [10000*n for n in range(1,len(balance))]
    x.append(300982)
    font1 = {
    'weight' : 'bold',
    'size'   : 48,
    }
    plt.xlabel('Request times', font1,position = (0.5,0.0))
    plt.ylabel('Traffic offloading', font1)
   
    plt.ylim(ymin=0, ymax=60)
    plt.xlim(xmin=x[0]-1000,xmax =x[-1]+1000) 
    #fig.align_labels()
    ax = plt.gca()
    #ax.set_xscale('log')   
    ax.yaxis.set_ticks_position('right')
    plt.minorticks_off()
    #ax.spines['right'].set_color('none')
    #names = ['10000','20000','30000','40000','50000','60000','70000','80000','']
    #names = ['','1%(300)','','','','','5%(1500)','','','10%(3000)','','','20%(6000)','','50%(15000)']   
    #ax.xaxis.set_major_locator(AutoLocator())
    #ax.xaxis.set_major_formatter(ScalarFormatter())
    #plt.minorticks_on()
    #plt.xticks([], [])
    plt.tick_params(which='both',direction='in')
    plt.tick_params(which='major',width=5,length=8)
    plt.tick_params(labelsize=40)
    #ax.set_xticklabels([])
    
    #plt.xticks(x, size = 10,rotation=0,verticalalignment ='bottom',position = (0,-0.06))
    plt.yticks(np.arange(0, 61, 10),['','10%','20%','30%','40%','50%','60%'], size = 40,)
    
    plt.subplots_adjust(bottom=0.115,right=0.930,left=0.070,top=0.980)

   
    plt.plot(x, privacy, '#ff93ff', label='Privacy-only',    linewidth=5, marker='o', markersize=5, )
    plt.plot(x, efficiency,'#82f479' , label='Efficiency-only', linewidth=5, marker='v', markersize=5, )
    plt.plot(x, balance, '#0edef1', label='Balance',      linewidth=5, marker='*', markersize=5, markerfacecolor = 'none')
    plt.plot(x, noneCache, '#ffcc60', label='None',        linewidth=5, marker='d', markersize=5, )
    #plt.plot(x, random, '#87bbea', label='Random',        linewidth=5, marker='X', markersize=5, ) 
    plt.plot(x, POC, '#a4a4a4', label='POC',  linewidth=5, marker="P", markersize=5, )
    plt.plot(x, Deepcache, '#ff8c80', label='DPC',  linewidth=5, marker="s", markersize=5, )
    #plt.plot(x, POC, '#a4a4a4', label='POC',linewidth=8, marker='s', markersize=25, markerfacecolor = 'none',linestyle = '-.')
    
    
    font2 = {
    'weight' : 'normal',
    'size'   : 28,
    }
    handles, labels = ax.get_legend_handles_labels() 
    plt.legend(handles[::-1], labels[::-1],prop = font2, loc='best', ncol=4, framealpha=1,frameon=True,facecolor ='w',columnspacing=0.8,)   

    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading-r3009.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading_1017-r3009.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/traffic_offloading_1017-r3009.jpeg')
    plt.show()
