# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'

import copy
from queue import PriorityQueue

import pandas as pd
import numpy as np

from Item import Item
from Event import Event
from Hypercube import Hypercube
from static import *

from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import time

import collections
import copy




class Popcache:
    """
    dim: 维度
    item_list: 视频列表，主键为视频的id
    hypercube_array: 立方空间的dim维数组
    max_day: 最大天数
    cur_day: 当天天数
    all_UITs: np.adarray 从pandas转化来的原始值
    event_lastday_list: 前一天的事件缓存列表
    event_count: 事件计数
    cache_size: 缓存列表大小
    cache_list: 缓存列表，存储视频id
    cache_set:

    if_disp: 是否展示
    """
    def __init__(self, dim, max_item_id, max_day, cur_day, all_UITs, cache_size, if_disp):
        """
        dim: 维度
        max_item_id: 最大视频id
        max_day: 最大天数
        cur_day: 当天天数
        all_UITs: np.adarray 从pandas转化来的原始值
        event_lastday_list: 前一天的事件缓存列表
        cache_size: 缓存列表大小

        if_disp: 是否展示
        """
        self.dim = dim
        self.item_list = [Item(item_id) for item_id in range(max_item_id)]
        self.hypercube_array = self.init_hypercube_array(dim)
        self.max_day = max_day
        self.cur_day = cur_day
        self.all_UITs = all_UITs
        self.curday_values = None
        self.event_lastday_list = []
        self.event_count = -1
        self.cache_size = cache_size
        self.cache_list = PriorityQueue()
        self.cache_set = set()

        self.if_disp = if_disp

    def init_hypercube_array(self, dim):
        """
        构建hypercube空间
        :param dim:
        :return:
        """
        hypercube_array = [Hypercube() for _ in range(2)]
        for _ in range(dim - 1):
            hypercube_array = [copy.deepcopy(hypercube_array) for _ in range(2)]
        return hypercube_array

    def enumerate_hypercubes(self):
        hypercubes_list = self.hypercube_array
        for depth in range(self.dim-1):
            tmp_list = []
            for element in hypercubes_list:
                tmp_list.extend(element)
            hypercubes_list = tmp_list

        if self.if_disp:
            print("enumerate {} depth, total length {}".format(dim, hypercubes_list.__len__()))

        return hypercubes_list

    def get_one_day_values(self):
        """
        获取一天的ndarray
        :return:
        """
        filter_UITs = self.all_UITs[self.all_UITs[:, 3] == self.cur_day]

        if self.if_disp:
            print("cur day {}".format(self.cur_day), end=" , ")
            print("total events {}".format(filter_UITs.shape[0]))
        return filter_UITs

    def extract_feature(self, item_id):
        """
        通过item_id获取视频的特征
        :param item_id:
        :return:
        """
        if self.cur_day > self.max_day:
            raise Exception("days exceed")

        cur_item = self.item_list[item_id]
        if np.sum(cur_item.history) == 0:
            return np.zeros((self.dim,), dtype=np.int32)

        feature = [0] * self.dim
        for pos, feature_day in enumerate(feature_day_list):
            if self.cur_day >= feature_day:
                check = np.sum(cur_item.history[self.cur_day - feature_day:self.cur_day])
            else:
                check = np.sum(cur_item.history[0:self.cur_day])
            if check > 0:
                feature[pos] = 1

        if self.if_disp:
            print("get the feature: ", np.array(feature, dtype=np.int32))

        return np.array(feature, dtype=np.int32)

    def update_feature(self):
        """
        根据ndarray更新当天视频的特征值
        :return:
        """
        count = 0
        for value_line in self.curday_values:
            item_id = value_line[2]
            self.item_list[item_id].history[self.cur_day] += 1
            count += 1

        item_count = 0
        for item in self.item_list:
            if item.history[self.cur_day]:
                item.popularity = item.history[self.cur_day]
                item_count += 1

        if self.if_disp:
            print("total deal with {} events, update {} items".format(count, item_count))

        return item_count

    def update_hypercube_estimate_value(self):
        """
        根据event_list列表更新hypercube他们的MN值
        :return:
        """

        total_popularity = 0

        if self.if_disp:
            print("update hypercube M N")

        for event in self.event_lastday_list:
            video_id = event.item
            event_feature = self.extract_feature(video_id)
            cube = self.select_hypercube(event_feature)
            popularity_plus = cube.update_popularity(self.item_list[video_id].popularity)
            total_popularity += popularity_plus

            if self.if_disp:
                print("current hypercube ", event_feature, end=" , ")
                print("popularity plus {}".format(popularity_plus))

        return total_popularity

    def update_cache_set(self):
        tmp_list = PriorityQueue()
        while not self.cache_list.empty():
            (old_priority, old_item) = self.cache_list.get()
            event_feature = self.extract_feature(old_item)
            cube = self.select_hypercube(event_feature)
            update_estimate_popularity = cube.get_popularity()
            tmp_list.put((update_estimate_popularity, old_item))

        if tmp_list.qsize() != self.cache_set.__len__():
            raise Exception("update heap error")

        self.cache_list = tmp_list

        for event in self.event_lastday_list:
            if event.item in self.cache_set:
                continue

            if len(self.cache_set) < self.cache_size:
                self.cache_set.add(event.item)
                self.cache_list.put((event.esti_popularity, event.item))
            else:
                (top_priority, top_item) = self.cache_list.queue[0]

                if event.esti_popularity >= top_priority:  # 替换条件
                    (replace_popularity, replace_item) = self.cache_list.get()
                    self.cache_set.remove(replace_item)
                    self.cache_list.put((event.esti_popularity, event.item))
                    self.cache_set.add(event.item)
                    # print("{} out and {} in".format(replace_popularity, event.esti_popularity))
        return True

    def update_one_day(self):
        """
        天数加1
        :return:
        """
        self.cur_day += 1
        self.event_lastday_list = []
        if self.cur_day > self.max_day:
            raise Exception("days exceed")

        if self.if_disp:
            print("one day adances, current day {}".format(self.cur_day))

        return self.cur_day

    def select_hypercube(self, feature):
        """
        根据视频的特征选择对应的hypercube
        :param feature:
        :return:
        """
        hypercube = self.hypercube_array
        for i in feature:
            hypercube = hypercube[i]

        assert isinstance(hypercube, Hypercube)

        if self.if_disp:
            print("current hypercube: ", feature)

        return hypercube

    def estimate_popularity(self, hypercube, event):
        """
        根据hypercube的总体估计当前事件的流行度
        :param hypercube:
        :param event_id:
        :return:
        """
        hypercube.add_event(event)       # 添加一个事件
        popularity = hypercube.get_popularity()     # 获取popularity
        event.esti_popularity = popularity

        if self.if_disp:
            print("the esti_popularity is {}".format(popularity))

        return popularity

    def event_add_oneday(self):
        """
        添加cur_day当天的event进event_list
        """
        self.curday_values = self.get_one_day_values()
        for value_line in self.curday_values:
            self.event_count += 1
            event = Event(id=value_line[0], user=value_line[1], item=value_line[2], occur_time=value_line[3], esti_popularity=0)
            self.event_lastday_list.append(event)

        if self.if_disp:
            print("today add {} events, current event pos: {}".format(self.curday_values.shape[0], self.event_count))    # for check use

        return self.event_lastday_list

    def curday_event_into_cube(self, is_validate=False):
        count = 0
        hit = 0
        for event in self.event_lastday_list:

            if self.if_disp:
                print("estimate event {}".format(event.id))

            event_feature = self.extract_feature(event.item)
            cube = self.select_hypercube(event_feature)
            _estimate_popularity = self.estimate_popularity(cube, event)

            hit += self.metric(event)
            count += 1

        if self.if_disp:
            print("total estimate {} events".format(count))

        if is_validate:
            print("today hit rate is {}".format(round(hit/count, 3)))

        return count, hit

    def print_cubes(self):
        print("day {} show cubes".format(self.cur_day))
        hypercubes_list = self.enumerate_hypercubes()

        for pos, hypercube in enumerate(hypercubes_list):
            print("current cube {}".format(pos), end=" , ")
            print("M:{} N:{}".format(hypercube.M, hypercube.N), end=" , ")
            print(hypercube.get_popularity())

    def metric(self, event):
        """
        最小堆比较，缓存击中返回1，否则返回0；
        :param event:
        :return:
        """
        hit = 0
        if event.item not in self.cache_set:
            # if len(self.cache_set) < self.cache_size:
            #     self.cache_set.add(event.item)
            #     self.cache_list.put((event.esti_popularity, event.item))
            # else:
            #     (top_priority, top_item) = self.cache_list.queue[0]
            #
            #     if event.esti_popularity >= top_priority:  # 替换条件
            #         (replace_popularity, replace_item) = self.cache_list.get()
            #         self.cache_set.remove(replace_item)
            #         self.cache_list.put((event.esti_popularity, event.item))
            #         self.cache_set.add(event.item)
                    # print("{} out and {} in".format(replace_popularity, event.esti_popularity))
            return hit
        else:
            hit = 1
            return 1


class ENV(object):
    def __init__(self,userNum,contentNum):
        self.userNum = userNum
        self.contentNum =contentNum

        self.r = np.zeros(shape=(userNum,contentNum),dtype=int)
        self.p = np.full(shape=contentNum,fill_value = 1/userNum)
        self.e = np.zeros(shape=contentNum)
        self.S = np.ones(shape=contentNum,dtype=int)
        self.l_edge = 0.1
        self.l_cp = 1

        self.B = np.full(shape=userNum,fill_value=15,dtype=int)

        self.pipe = collections.OrderedDict()


    #有序字典实现LRU
    def updateEgdeCache(self,action,t):
        edgeHit = 0
        for i in np.argwhere(action==1).squeeze(-1):
            if i in self.pipe.keys():
                self.pipe.pop(i)
                edgeHit = 1
            elif len(self.pipe) >= 500:
                self.e[self.pipe.popitem(last=False)[0]] = 0
            self.pipe[i] = t
            self.e[i] = 1
        return edgeHit
    
    def updateEnv(self,u,action,t):
        
        p_tmp = ((self.r[u] | action)-self.r[u])*(1/self.userNum) + self.p
        self.p = np.where(p_tmp<1-1/self.userNum,p_tmp,1-1/self.userNum)

        self.r[u] = self.r[u] | action

        return self.updateEgdeCache(action,t)

    def getStatus(self):
        return (torch.from_numpy(self.r),
                torch.from_numpy(self.p) , 
                torch.from_numpy(self.e),
                torch.from_numpy(self.S),
                self.l_edge,
                self.l_cp)

    #def reset(self):
    #    self.r = np.zeros(shape=(self.userNum,self.contentNum),dtype=int)
    #    self.p = np.full(shape=self.contentNum,fill_value = 1/self.userNum)
    #    self.e = np.zeros(shape=self.contentNum)
    #    self.S = np.ones(shape=self.contentNum,dtype=int)
    #    self.l_edge = 0.1
    #    self.l_cp = 1
    #    self.B = np.full(shape=self.userNum,fill_value=15,dtype=int)
    #    self.pipe = collections.OrderedDict()
class UE(object):
    def __init__(self,u,env,rewardPara):
        self.u = u

        self.W = []
        self.v = torch.zeros(size=(env.contentNum,),dtype=int)

        self.Bu = int(env.B[self.u])
        self.contentNum = env.contentNum
        self.userNum = env.userNum

        self.r , self.p , self.e, self.S,self.l_edge, self.l_cp = env.getStatus()

        self.action = torch.zeros(size=(env.contentNum,),dtype=int)
        self.lastAction = self.action

        self.reward = 0
        self.ALPHAh = rewardPara['alpha']
        self.BETAo =  rewardPara['betao']
        self.BETAl =  rewardPara['betal']

        self.edgeHit = 0

    def updateViewContent(self,i):
        self.W.append(i)
        self.v[i] = 1

    
    def getReward(self,lastru,lastp,ru,p,i,action,S,Bu,l_edge,l_cp,e):

        self.Rh = - self.ALPHAh * (torch.log(lastru * lastp + (1-lastru) * (1-lastp)).sum() - torch.log(ru * p + (1-ru) * (1-p)).sum())

        self.Ro =   self.BETAo * action[i] * (S[i] / Bu + ( e[i] * l_edge + ( 1-e[i] ) * l_cp ) / S[i])

        self.Rl =   self.BETAl * ( ( 1 - action[i] )  * ( l_cp - ( e[i] * l_edge + ( 1 - e[i] ) * l_cp ) ) ) / S[i]

        #self.Rh[i] = self.Rh[i] + self.Ro + self.Rl

        return  self.Rh+self.Ro+self.Rl

    def selectAction(self,env,uit,popcache):

        self.lastAction = self.action
        self.lastp = self.p
        self.lastr = self.r

        self.updateViewContent(uit[1])
        self.r , self.p , self.e, self.S, self.l_edge, self.l_cp = env.getStatus()
        
        self.reward = self.getReward(self.lastr[self.u],self.lastp,self.r[self.u],self.p,self.W[-1],self.lastAction,self.S,self.Bu,self.l_edge,self.l_cp,self.e)
        
        cacheSet = popcache.cache_set.copy()
        
        self.action = torch.zeros(size=(env.contentNum,),dtype=int)
        self.action[self.W[-1]] = 1
        if self.W[-1] not in cacheSet and len(cacheSet)>0:
            cacheSet.pop() 
        for i in cacheSet:
            self.action[i] = 1

        self.edgeHit += env.updateEnv(self.u,self.action.numpy(),uit[2])

        #print(self.W[-1],np.argwhere(ue.action.numpy()==1),cacheSet)
        return self.action


level = 'u'
# flag = "不按天更新"
#data_path = '/home/zhangxz/workspace/data/Nati1000_U500_V30/'
#data_path = '/home/ubuntu/data/dataset/R1584_U50_V2/'
#data_path = '/home/ubuntu/data/dataset/R3009_U5_V100/'
#part_one = pd.read_csv(data_path + 'train.csv', header=None)
#part_two = pd.read_csv(data_path + 'test.csv', header=None)
#UIT = pd.concat([part_one, part_two], axis=0)
#UIT = pd.read_csv(data_path + 'UIT.csv')

#import os
data_path = '/home/ubuntu/data/dataset/R3009_U5_V100_header/'
#if not os.path.exists(data_path):
#    os.makedirs(data_path)

#UIT.to_csv(data_path + 'UIT.csv',index=0,header=None)
UIT = pd.read_csv(data_path + 'UIT.csv',header=None)

#UIT = UIT[UIT['day']<18]
group_count = 0
result_list = []
#UIT.columns = [[0,1,2,3,4,5,6,7,8,9,10,11]]

trainUIT = UIT[UIT[2]<18]
contentNum = max(UIT[1])+1
userNum = max(UIT[0])+1
UIT[2] = UIT[3]//(60)


all_UITs = UIT.values
items_seq = max(all_UITs[:, 1] + 1)
index = np.arange(0, all_UITs.shape[0]).reshape((all_UITs.shape[0], 1))
all_UITs = np.column_stack((index, all_UITs))

trainT = 18*24
testT = 12*24
days = trainT + testT

UEs = {}
rewardPara = {"alpha":1,"betao":0.5,"betal":0.5}
env = ENV(userNum,contentNum)
UEs = {}
sumReward = np.zeros(3)
loss = 0
UEHit = np.zeros(userNum)
edgeHit = 0
curDay = 0

cache_size_ratio =0.0015
cache_size = int(contentNum * cache_size_ratio)
popcache = Popcache(dim, max_item_id = contentNum, max_day=days, all_UITs=all_UITs,cur_day=curDay,cache_size=cache_size,if_disp=False)

for index,trace in UIT.iterrows():
    
    uit = trace.to_numpy()
    if uit[0] not in UEs:
        UEs[uit[0]] = UE(uit[0],env,rewardPara)
    ue = UEs[uit[0]]
    actionIndex = np.argwhere(ue.lastAction.numpy())
    if uit[1] in actionIndex:
        UEHit[uit[0]] += 1
    elif uit[1] in env.pipe.keys():
        edgeHit += 1

    ue.selectAction(env=env,uit=uit,popcache=popcache)
    if curDay<uit[2]:
        curDay += 1
        print(curDay)
        event_list = popcache.event_add_oneday()
        click, hit = popcache.curday_event_into_cube()
        popcache.update_feature()
        total_popularity = popcache.update_hypercube_estimate_value()
        popcache.update_cache_set()
        # popcache.print_cubes()
        # popcache.enumerate_hypercubes()
        popcache.update_one_day()

    sumReward[0] += float(ue.Rh)
    sumReward[1] += float(ue.Rl)
    sumReward[2] += float(ue.Ro)
    if index % 10000 == 0 :
        psi = 0
        p = torch.from_numpy(env.p)
        for u in UEs:
            psi += torch.log(UEs[u].r[u] * p + (1-UEs[u].r[u]) * (1-p)).sum() / torch.log(UEs[u].v * p + (1-UEs[u].v) * (1-p)).sum()
        print("--Time:",time.asctime( time.localtime(time.time())),"  Index:",index,"  Loss:",round(loss/(index+1),5),"--")
        print("Reward:",np.around(sumReward/(index+1),5),"total reward:",round(sumReward.sum()/(index+1),5))
        print("UEHitrate:",round(UEHit.sum()/(index+1),5)," edgeHitrate",round(edgeHit/(index+1),5),"sumHitrate",round((edgeHit+UEHit.sum())/(index+1),5)," privacy:",round(float(psi)/len(UEs),5))
        print()
psi = 0
p = torch.from_numpy(env.p)
for u in UEs:
    psi += torch.log(UEs[u].r[u] * p + (1-UEs[u].r[u]) * (1-p)).sum() / torch.log(UEs[u].v * p + (1-UEs[u].v) * (1-p)).sum()
print()
print("----------------------------------------------------------------")
print("--Time:",time.asctime( time.localtime(time.time())),"  Index:",index,"  Loss:",round(loss/(index+1),5),"--")
print("Reward:",np.around(sumReward/(index+1),5),"total reward:",round(sumReward.sum()/(index+1),5))
print("UEHitrate:",round(UEHit.sum()/(index+1),5)," edgeHitrate",round(edgeHit/(index+1),5),"sumHitrate",round((edgeHit+UEHit.sum())/(index+1),5),"privacy:",round(float(psi)/len(UEs),5))
print("----------------------------------------------------------------")
print()