import cv2
import numpy as np
from math import ceil
import time
from cscrollgrid import fast_min_max

FLOATMAX = 1000000.0

class ScrollGrid(object):
    def __init__(self, 
                 resolution, 
                 crop_range,
                 ):
        '''
        resoluation: grid size in meter
        range: (xmin, xmax, ymin, ymax) in meter, 
               the cropped grid map covers [(xmin, ymin), (xmax, ymax))
        
        '''

        self.resolution = resolution
        self.xmin, self.xmax, self.ymin, self.ymax  = crop_range
        # the (xmin, ymin) is located at the center of a grid
        # so if the (center_pt - min_pt) can be devided by reso, the center_pt will also be located at the center of a grid
        self.xnum = int(ceil((self.xmax - self.xmin - self.resolution/2.0)/self.resolution)) + 1 
        self.ynum = int(ceil((self.ymax - self.ymin - self.resolution/2.0)/self.resolution)) + 1

        self.emem = np.zeros((self.xnum, self.ynum, 2), dtype=np.float32) # heightmap
        self.cmem = np.zeros((self.xnum, self.ynum, 3), dtype=np.uint8) # rgbmap

        print('Map initialized, resolution {}, range {}, shape {}'.format(self.resolution, crop_range, (self.xnum, self.ynum)))

        self.initialize_elevation_map()

    def initialize_elevation_map(self):
        self.emem[:, :, 0].fill(FLOATMAX) # store min-height
        self.emem[:, :, 1].fill(-FLOATMAX) # store max-height
        self.cmem.fill(0) # rgbmap

    def pc_xy_to_grid_ind(self, x, y):
        xind = np.round((x - self.xmin)/self.resolution)
        yind = np.round((y - self.ymin)/self.resolution)
        return xind, yind

    def pc_coord_to_grid_ind(self, pc_coord):
        return np.round((pc_coord-np.array([self.xmin, self.ymin]))/self.resolution).astype(np.int32)

    def pc_to_map(self, points, colors=None):
        starttime = time.time()
        # import ipdb; ipdb.set_trace()
        grid_inds = self.pc_coord_to_grid_ind(points[:, 0:2])
        self.initialize_elevation_map()

        grid_inds = grid_inds.astype(np.uint16)
        zgrid = points[:, 2].astype(np.float32)
        self.emem, self.cmem = fast_min_max(self.emem, self.cmem, grid_inds, zgrid, colors, self.xnum, self.ynum)

        # for i, ind in enumerate(grid_inds):
        #     if ind[0]>=0 and ind[0]<self.xnum and ind[1]>=0 and ind[1]<self.ynum:
        #         if points[i, 2] < self.emem[ind[0], ind[1], 0]:
        #             self.emem[ind[0], ind[1], 0] = points[i, 2]
        #         if points[i, 2] > self.emem[ind[0], ind[1], 1]:
        #             self.emem[ind[0], ind[1], 1] = points[i, 2]
        #             if colors is not None:
        #                 self.cmem[ind[0], ind[1], :] = colors[i,:]

        print('Localmap convert time: {}'.format(time.time()-starttime))
        # import ipdb;ipdb.set_trace()
        # self.show_heightmap()

    def show_heightmap(self, hmin=-1, hmax=4):
        mask = self.emem[:,:,0]==FLOATMAX
        disp1 = np.clip((self.emem[:, :, 0] - hmin)*100, 0, 255).astype(np.uint8)
        disp2 = np.clip((self.emem[:, :, 1] - hmin)*100, 0, 255).astype(np.uint8)
        disp1[mask] = 0
        disp2[mask] = 0
        disp = np.concatenate((cv2.flip(disp1, -1), cv2.flip(disp2, -1)) , axis=1)
        # disp = cv2.resize(disp, (0, 0), fx=2., fy=2.)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        cv2.imshow('height',disp_color)
        cv2.waitKey(1)

    def show_colormap(self):
        disp = cv2.flip(self.cmem, -1)
        # disp = cv2.resize(disp, (0, 0), fx=2., fy=2.)

        cv2.imshow('color',disp)
        cv2.waitKey(1)

    def get_height(self, x, y):
        pass


    def get_heightmap(self):
        return self.emem

    def get_rgbmap(self):
        return self.cmem

if __name__ == '__main__':
    localmap = ScrollGrid(0.01, (1., 5., -2., 2.))
    points = np.random.rand(1000000, 3)
    points[:, 0] = points[:, 0] *3
    points[:, 1] = points[:, 1] *4 -2
    colors = np.random.rand(1000000, 3)
    colors = (colors * 256).astype(np.uint8)
    localmap.pc_to_map(points, colors)
    localmap.show_heightmap()
    localmap.show_colormap()
