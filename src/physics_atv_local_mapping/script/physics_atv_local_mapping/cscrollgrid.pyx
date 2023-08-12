import numpy as np
cimport numpy as np
np.import_array()

# from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, int32_t #, uint16_t, uint32_t, uint64_t

# cdef extern void get_depth_parallel(int32_t* X, int32_t* Y, int32_t* Z, uint32_t height, uint32_t width, int32_t* gids, uint32_t* emap, uint32_t g0, uint32_t g1, uint8_t k_threads)

# cdef extern void c_fast_min_max(
#     int16_t* mem, float32_t m1, uint32_t m2,
#     uint16_t* mem_indices, uint32_t mi1,
#     int32_t* z_gid, float32_t nPoints)

cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
def fast_min_max(np.ndarray[np.float32_t, ndim=3] emem,
                 np.ndarray[np.uint8_t, ndim=3] cmem,
                 np.ndarray[np.uint16_t, ndim=2] mem_indices,
                 np.ndarray[np.float32_t, ndim=1] z_gid, 
                 np.ndarray[np.uint8_t, ndim=2] c_gid,
                 int32_t xnum, int32_t ynum):

    cdef unsigned int nPoints = mem_indices.shape[0]

    '''
    c_fast_min_max(
        <int16_t*> mem.data, mem.shape[1], mem.shape[2],
        <uint16_t*> mem_indices.data, mem_indices.shape[1],
        <int32_t*> z_gid.data, nPoints
    )
    return
    '''

    cdef unsigned int i
    cdef unsigned int x, y
    cdef float z

    for i in range(nPoints):
        x = <unsigned int>mem_indices[i, 0]
        y = <unsigned int>mem_indices[i, 1]
        z = <float>z_gid[i]

        if x >= 0 and x<xnum and y>=0 and y<ynum:

            if z < emem[x, y, 0]:
                emem[x, y, 0] = z

            if z > emem[x, y, 1]:
                emem[x, y, 1] = z
                # cmem[x, y, :] = c_gid[i]
                cmem[x, y, 0] = <uint8_t>c_gid[i,0]
                cmem[x, y, 1] = <uint8_t>c_gid[i,1]
                cmem[x, y, 2] = <uint8_t>c_gid[i,2]

    return emem, cmem

@cython.wraparound(False)
@cython.boundscheck(False)
def voxel_filter(np.ndarray[np.uint16_t, ndim=2] pt_indices,
                 np.ndarray[np.float32_t, ndim=1] pt_dist, 
                 int32_t xnum, int32_t ynum, int32_t znum):

    cdef unsigned int nPoints = pt_indices.shape[0]

    cdef np.ndarray mindist = np.ones((xnum, ynum, znum), dtype=np.float32) * 10000
    cdef np.ndarray mininds = np.ones((xnum, ynum, znum), dtype=np.int32)
    cdef np.ndarray resmask = np.zeros((nPoints), dtype=np.bool)

    cdef unsigned int i
    cdef unsigned int x, y, z

    for i in range(nPoints):
        x = <unsigned int>pt_indices[i, 0]
        y = <unsigned int>pt_indices[i, 1]
        z = <unsigned int>pt_indices[i, 2]

        if x >= 0 and x<xnum and y>=0 and y<ynum and z>=0 and z<znum:

            dist = pt_dist[i]
            if dist < mindist[x, y, z]:
                if mindist[x, y, z] < 9999:
                    resmask[mininds[x, y, z]] = False
                mindist[x, y, z] = dist
                mininds[x, y, z] = i
                resmask[i] = True

    return resmask
