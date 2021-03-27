# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 22:07
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'nyu':
            return '/home/cap6412.student27/Phase3/data/nyudepthv2'
        elif database == 'kitti':
            return '/home/cap6412.student27/Phase3/data/kitti'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
