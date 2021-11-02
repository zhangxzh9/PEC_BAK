# -*- coding: utf-8 -*-

"""
note something here
static path
"""

__author__ = 'Wang Chen'
__time__ = '2019/7/15'

# 100 user
# users_num = 100
# data_path = 'D:\\workData\\tecent_for_hawkes\\data{}\\'.format(users_num)
# train_file = data_path + 'train.csv'
# test_file = data_path + 'test.csv'

# dim = 3  # 1,3,7
# days = 30
# feature_day_list = [1,3,7]
# items_num = 2169



# 1000 user
# users_num = 1000
# data_path = 'D:\\workData\\tecent_for_hawkes\\data{}\\'.format(users_num)
# train_file = data_path + 'train.csv'
# test_file = data_path + 'test.csv'

# dim = 3  # 1,3,7
# days = 30
# feature_day_list = [1,3,7]
# items_num = 13095




# 10000 user
# users_num = 10000
# data_path = 'D:\\workData\\tecent_for_hawkes\\data{}\\'.format(users_num)
# train_file = data_path + 'train.csv'
# test_file = data_path + 'test.csv'
#
# dim = 4  # 1,3,7
# days = 30
# feature_day_list = [1,3,7,21]
# items_num = 56043


# # new data
# data_path = 'D:\\workData\\tecent_for_hawkes\\sample10000\\'
# train_file = data_path + 'train.csv'
# test_file = data_path + 'test.csv'

# dim = 4  # 1,3,7
# days = 30
# feature_day_list = [1,3,7,21]
# items_num = 10000

# final data
#data_path = '/home/appvin/cache/sample_info_items/sample_train10000/'
#train_file = data_path + 'label_train.csv'
#test_file = data_path + 'label_test.csv'

#dim = 4  # 1,3,7
#days = 30
#feature_day_list = [1,3,7,21]
#items_num = 10000

dim = 4  # 1,3,7
days = 24*30*60
feature_day_list = [5*60*60,30*60,5*24*60,20*24*60]
#items_num = 10000