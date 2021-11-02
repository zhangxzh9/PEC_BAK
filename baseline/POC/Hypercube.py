# -*- coding: utf-8 -*-

"""
note something here

"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'


class Hypercube:
    """
    event_list: 事件列表
    event_dict: 记录每个事件的列表位置
    event_num: 事件数量
    event_total_popu: 事件总的流行度
    """

    def __init__(self):
        self.event_list = []
        self.event_dict = {}
        self.N = 0          ###  event_num
        self.M = 0       ###  event_total_popu
        self.count = -1   ### 事件数量记录

    def get_popularity(self):
        """
        get the popularity for the hypercube
        M: sum of popularity for all events
        N: sum of the number of events
        :return: the average popularity of the hypercube
        """
        if self.M == 0:      ### 如果事件总的popularity为0，这里初始化一个参数
            return 0
        else:
            return round((self.M / self.N), 3)

    def update_popularity(self, M):
        """
        once a day update the M and N
        we need to update the popularity for item then calculate the mean popularity for the hypercube
        :return:
        """
        self.M += M
        self.N += 1
        return M


    def add_event(self, event):
        self.event_list.append(event)
        self.count += 1
        self.event_dict[event.id] = self.count
        return True

    def select_event(self, event_id):
        return self.event_dict[event_id]


if __name__ == '__main__':
    pass