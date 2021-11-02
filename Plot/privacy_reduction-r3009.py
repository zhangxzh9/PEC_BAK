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
    #random       = list(np.array([3.87249, 3.39743, 3.05268, 2.80303, 2.58978, 2.#42809,
    #   2.29468, 2.16831, 2.05126, 1.93582, 1.83487, 1.7502 , 1.66939,
    #   1.59503, 1.52718, 1.46064, 1.40018, 1.33967, 1.28351, 1.23272,
    #   1.18247, 1.13662, 1.09053, 1.04733, 1.00647, 0.96889, 0.93318,
    #   0.8988 , 0.86455, 0.83082]))
    efficiency   = list(np.array([1.76804, 1.5731 , 1.50829, 1.46777, 1.43907, 1.41072, 1.38747,
       1.36871, 1.34893, 1.32802, 1.31626, 1.30422, 1.2918 , 1.28065,
       1.27051, 1.25932, 1.24938, 1.24513, 1.23932, 1.23581, 1.22904,
       1.22155, 1.21526, 1.20874, 1.20457, 1.19995, 1.19582, 1.19184,
       1.18892, 1.1866 , 1.18627]))
    privacy      = list(np.array([0.88683, 0.87111, 0.87353, 0.87906, 0.87733, 0.88447, 0.8922 ,
       0.89272, 0.89315, 0.88667, 0.88222, 0.88747, 0.88586, 0.88879,
       0.89329, 0.89496, 0.89571, 0.88806, 0.88629, 0.88298, 0.86463,
       0.86749, 0.87067, 0.87345, 0.87591, 0.87812, 0.8805 , 0.8826 ,
       0.88436, 0.88559, 0.88569]))
    Deepcache    = list(np.array([ 3.94465, 3.15531, 2.77583, 2.5636 , 2.4029 , 2.29392, 2.2119 ,
       2.14106, 2.07378, 1.99797, 1.94271, 1.90889, 1.86838, 1.84176,
       1.82212, 1.7953 , 1.77474, 1.74094, 1.71988, 1.69583, 1.64948,
       1.63865, 1.62586, 1.6169 , 1.60787, 1.59832, 1.58801, 1.57983,
       1.57357, 1.56894, 1.56854]))
    POC = list(np.array([ 3.99454, 3.31415, 2.92165, 2.70939, 2.5472 , 2.43707, 2.35527,
       2.28329, 2.21089, 2.13217, 2.07445, 2.03621, 1.9926 , 1.96232,
       1.9403 , 1.9126 , 1.89041, 1.85584, 1.83394, 1.81074, 1.76617,
       1.75163, 1.73536, 1.7235 , 1.71307, 1.702  , 1.68867, 1.67937,
       1.67154, 1.6659 , 1.6653 ]))
    noneCache    = list(np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))
    balance = list(np.array([0.88866, 0.92239, 0.8483 , 0.78819, 0.77447, 0.77967, 0.78902,
       0.79672, 0.80597, 0.81528, 0.82163, 0.82698, 0.83159, 0.83352,
       0.83749, 0.84154, 0.84653, 0.84831, 0.85106, 0.85429, 0.85917,
       0.86258, 0.86518, 0.86651, 0.8697 , 0.87188, 0.87435, 0.87604,
       0.87879, 0.88124, 0.88132]))

    x = [10000*n for n in range(1,len(balance))]
    x.append(300982)
    font1 = {
    'weight' : 'bold',
    'size'   : 48,
    }
    plt.xlabel('Request times', font1,position = (0.5,0.0))
    plt.ylabel('Privacy reduction', font1)
   
    plt.ylim(ymin=0, ymax=4)
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
    
    #plt.xticks(x, size = 40,rotation=0,verticalalignment ='bottom',position = (0,-0.06))
    plt.yticks(np.arange(0.5, 4.1, 0.5), size = 40)
    
    plt.subplots_adjust(bottom=0.115,right=0.930,left=0.070,top=0.980)

   
    plt.plot(x, privacy, '#ff93ff', label='Privacy-only',    linewidth=5, marker='o', markersize=7, )
    plt.plot(x, efficiency,'#82f479' , label='Efficiency-only', linewidth=5, marker='v', markersize=7, )
    plt.plot(x, balance, '#0edef1', label='Balance',      linewidth=5, marker='*', markersize=7, markerfacecolor = 'none')
    plt.plot(x, noneCache, '#ffcc60', label='None',        linewidth=5, marker='d', markersize=7, )
    #plt.plot(x, random, '#87bbea', label='Random',        linewidth=5, marker='X', markersize=7, ) 
    plt.plot(x, POC, '#a4a4a4', label='POC',  linewidth=5, marker="P", markersize=7, )
    plt.plot(x, Deepcache, '#ff8c80', label='DPC',  linewidth=5, marker="s", markersize=7, )
    #plt.plot(x, POC, '#a4a4a4', label='POC',linewidth=8, marker='s', markersize=25, markerfacecolor = 'none',linestyle = '-.')
    
    font2 = {
    'weight' : 'normal',
    'size'   : 36,
    }
    handles, labels = ax.get_legend_handles_labels() 
    plt.legend(handles[::-1], labels[::-1],prop = font2, loc='best', ncol=2, framealpha=1,frameon=True,facecolor ='w',columnspacing=0.8,)   

    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction-r3009.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction_1017-r3009.eps')
    plt.savefig('/home/ubuntu/data/PEC/Plot/output/privacy_reduction_1017-r3009.jpeg')
    plt.show()
