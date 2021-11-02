# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'


class Event:
    """
    id: 事件序号
    user: 用户id
    item: 视频id
    occur_time: 发生时间
    esti_popularity: 估计的流行度
    """

    def __init__(self, id, user, item, occur_time, esti_popularity=0):
        self.id = id
        self.user = user
        self.item = item
        self.occur_time = occur_time
        self.esti_popularity = esti_popularity




if __name__ == '__main__':
    pass