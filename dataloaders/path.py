class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'nyu':
            return '/home/dsilva/MonoDepthAttacks/data/nyu_depth_v2'
        elif database == 'kitti':
            return '/home/dsilva/MonoDepthAttacks/data/kitti'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
