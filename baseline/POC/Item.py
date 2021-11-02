# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'


import numpy as np


class Item:
    """
    id              # 视频的id
    history         # 观看向量，这里简单记为0 1 向量
    popularity      # 当天的流行度，每天更新
    """
    def __init__(self, item_id):
        self.item_id = item_id
        #self.history = np.array([0]*30, np.int32)
        self.history = np.array([0]*30*24, np.int32)
        self.popularity = 0


if __name__ == '__main__':
    pass
