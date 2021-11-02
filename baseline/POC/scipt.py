# -*- coding: utf-8 -*-


"""
note something here
script for algo Popcache
"""

__author__ = 'Wang Chen'
__time__ = '2019/7/14'


import copy
from queue import PriorityQueue

import pandas as pd
import numpy as np

from popcache.Item import Item
from popcache.Event import Event
from popcache.Hypercube import Hypercube
from popcache.static import *


# demo for one day
UIT = pd.read_csv(data_path + 'train.csv', header=None)

def init_hypercube_array(d):
    hypercube_array = [Hypercube() for _ in range(2)]
    for _ in range(d - 1):
        hypercube_array = [copy.deepcopy(hypercube_array) for _ in range(2)]
    return hypercube_array


def input_one_day(day):
    return UIT[UIT[2] == day]


def extract_feature(item_id, item_list, cur_day, max_day):
    if cur_day > max_day:
        raise Exception("days exceed")

    cur_item = item_list[item_id]
    if np.sum(cur_item.history) == 0:
        return np.zeros((dim,), dtype=np.int32)

    feature = [0]*dim
    for pos, feature_day in enumerate(feature_day_list):
        if cur_day >= feature_day:
            check = np.sum(cur_item.history[cur_day-feature_day:cur_day])
        else:
            check = np.sum(cur_item.history[0:cur_day])

        if check > 0:
            feature[pos] = 1

    return np.array(feature, dtype=np.int32)


def update_feature():
    pass


def estimate_popularity(hypercube_array, feature):
    hypercube = hypercube_array
    for i in feature:
        hypercube = hypercube[i]

    assert isinstance(hypercube, Hypercube)

def concatenate():
    part_one = pd.read_csv(data_path + 'train.csv', header=None)
    part_two = pd.read_csv(data_path + 'test.csv', header=None).drop([7], axis=1)
    result = pd.concat([part_one, part_two], axis=0)
    return result


if __name__ == '__main__':
    # a = init_hypercube_array(3)
    # UIT = input_one_day(0)
    # print("hello popcache")
    #
    # item_list = [Item(item_id) for item_id in range(2169)]
    # b = extract_feature(2, item_list, 1, 30)
    # estimate_popularity(a, b)

    # q = PriorityQueue()
    #
    # # 格式：q.put((数字,值))
    # # 特点：数字越小，优先级越高
    # q.put((1, 'lori'))
    # q.put((-1, 'Jseon'))
    # q.put((10, 'King'))
    #
    # q.put((-1, 'wang'))

    df_con = concatenate()
    df_con_1 = df_con.drop_duplicates([0], inplace=False)
    df_con_2 = df_con.drop_duplicates([1], inplace=False)
