#include "ScrollGrid.h"
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <cstring>

#define FLOWMAX 1000000.0

ScrollGrid::ScrollGrid()
: colormap(nullptr),
  heightmap(nullptr),
  colormap_inflate(nullptr),
  heightmap_inflate(nullptr),
  point_count(nullptr),
  height_mean_sq(nullptr)
{

};

void ScrollGrid::init(double mapresolution, 
                       double x_min, double x_max, 
                       double y_min, double y_max)
{
    xmin = x_min;
    xmax = x_max;
    ymin = y_min;
    ymax = y_max;
    resolution = mapresolution;

    // now the map center is at the cross of four pixels, so does (xmin, ymin)
    xnum = ceil((xmax - xmin)/resolution); //please make sure the range can be devided by the resolution
    ynum = ceil((ymax - ymin)/resolution); //please make sure the range can be devided by the resolution
    gridnum = xnum * ynum;

    if(colormap)
        delete[] colormap;
    if(heightmap)
        delete[] heightmap;
    if(colormap_inflate)
        delete[] colormap_inflate;
    if(heightmap_inflate)
        delete[] heightmap_inflate;
    if(point_count)
        delete[] point_count;
    if(height_mean_sq)
        delete[] height_mean_sq;

    initialize_elevation_map();
    printf("Map initialized, resolution %f, range (%f, %f, %f, %f), shape (%d, %d)\n",resolution, x_min, x_max, y_min, y_max, xnum, ynum);

}


// ScrollGrid& ScrollGrid::operator=(const ScrollGrid &other)
// {
//     if(&other == this)
//         return *this;
//     std::cout<<"operator"<<std::endl;
//     xmin = other.xmin;
//     xmax = other.xmax;
//     ymin = other.ymin;
//     ymax = other.ymax;
//     resolution = other.resolution;

//     // now the map center is at the cross of four pixels, so does (xmin, ymin)
//     xnum = other.xnum; //please make sure the range can be devided by the resolution
//     ynum = other.ynum; //please make sure the range can be devided by the resolution
//     gridnum = other.gridnum;

//     if(colormap)
//         delete[] colormap;
//     if(minheightmap)
//         delete[] minheightmap;
//     if(maxheightmap)
//         delete[] maxheightmap;

//     initialize_elevation_map();
//     memcpy(colormap, other.colormap, gridnum*3*sizeof(uint8_t));
//     memcpy(minheightmap, other.minheightmap, gridnum*sizeof(double));
//     memcpy(maxheightmap, other.maxheightmap, gridnum*sizeof(double));
//     return *this;
// }


ScrollGrid::~ScrollGrid()
{
    if(colormap)
        delete[] colormap;
    if(heightmap)
        delete[] heightmap;
    if(colormap_inflate)
        delete[] colormap_inflate;
    if(heightmap_inflate)
        delete[] heightmap_inflate;
    if(point_count)
        delete[] point_count;
    if(height_mean_sq)
        delete[] height_mean_sq;
}

void ScrollGrid::initialize_elevation_map()
{
    colormap = new uint8_t[gridnum*3];
    heightmap = new float[gridnum*4]; // four channels: min, max, mean, std
    point_count = new int[gridnum];
    height_mean_sq = new float[gridnum];

    reset_elevation_map();
}

void ScrollGrid::reset_elevation_map()
{
    for(int i=0; i<gridnum; i++)
    {
        heightmap[i*4] = FLOWMAX; // first channel is minheight, second channel is maxheight
        heightmap[i*4 + 1] = -FLOWMAX;
        heightmap[i*4 + 2] = 0; // first channel is minheight, second channel is maxheight
        heightmap[i*4 + 3] = 0;
        colormap[i] = 0;
        colormap[i + gridnum] = 0;
        colormap[i + gridnum*2] = 0;

        point_count[i] = 0;
        height_mean_sq[i] = 0;
    }
}

int ScrollGrid::getMapHeight()
{
    return xnum;
}

int ScrollGrid::getMapWidth()
{
    return ynum;
}

// void ScrollGrid::pc_xy_to_grid_ind(const double x, const double y, int &xind, int &yind)
// {
//         xind = int((x - xmin)/resolution);
//         yind = int((y - ymin)/resolution);
// }

void ScrollGrid::pc_to_map(const pcl::PointCloud<PointT> &pc)
{ // assume the points are downsampled before calling this function

    // iterate through all the points
    // convert x y to xind, yind
    // compare the z value and update the color value
    auto start = std::chrono::steady_clock::now();

    // std::vector<PointT, Eigen::aligned_allocator<PointT> >::iterator it;
    int xind, yind, mapind, mapind_min, mapind_max, mapind_mean, mapind_std;
    reset_elevation_map();

    int count = 0;
    for(size_t idx = 0 ; idx < pc.points.size(); idx++ )
    {
        const PointT &pt = pc.points[idx];

        xind = int((pt.x - xmin)/resolution);
        yind = int((pt.y - ymin)/resolution);
        mapind = xind * ynum + yind; // convert 2d ind to 1d ind
        mapind_min = mapind * 4;
        mapind_max = mapind * 4 +1;
        mapind_mean = mapind * 4 +2;
        mapind_std = mapind * 4 +3;

        if(xind >= 0 && xind < xnum && yind >= 0 && yind < ynum) // point is within the range of the map
        {
            point_count[mapind] += 1;
            float cnt = float(point_count[mapind]);
            if(pt.z < heightmap[mapind_min])
                heightmap[mapind_min] = pt.z;
            if(pt.z > heightmap[mapind_max])
                heightmap[mapind_max] = pt.z;
            heightmap[mapind_mean] = (cnt-1)/(cnt) * heightmap[mapind_mean] + pt.z/cnt;
            height_mean_sq[mapind] = (cnt-1)/(cnt) * height_mean_sq[mapind] + pt.z*pt.z/cnt;
            heightmap[mapind_std] = height_mean_sq[mapind] - heightmap[mapind_mean] * heightmap[mapind_mean];

            colormap[mapind*3] = (cnt-1)/(cnt) * colormap[mapind*3] + pt.r/cnt;
            colormap[mapind*3 + 1] = (cnt-1)/(cnt) * colormap[mapind*3 + 1] + pt.g/cnt;
            colormap[mapind*3 + 2] = (cnt-1)/(cnt) * colormap[mapind*3 + 2] + pt.b/cnt;
            // std::cout<<pt<<std::endl;

            count += 1;
        }
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    // printf("PC to map convert in %.4f seconds, %d points in range. \n", elapsed_seconds.count(), count );
}

float* ScrollGrid::getHeightmap()
{
    return heightmap;
}

uint8_t* ScrollGrid::getColormap()
{
    return colormap;
}

float* ScrollGrid::getInflateHeightmap()
{
    return heightmap_inflate;
}

uint8_t* ScrollGrid::getInflateColormap()
{
    return colormap_inflate;
}


void ScrollGrid::show_height_map()
{
    uint8_t *heightmapdisp = new uint8_t[gridnum*4];
    int Scale = 100;
    double hmin=-1.0, hmax=4.0;
    int mapind, mapind_max, mapind_min, mapind_mean, mapind_std;
    int dispmin, dispmax, dispmean, dispstd;
    for(int i=0;i<xnum;i++)
    {
        for(int j=0;j<ynum;j++)
        {
            mapind = i*ynum + j;
            mapind_min = mapind * 4;
            mapind_max = mapind * 4 + 1;
            mapind_mean = mapind * 4 + 2;
            mapind_std = mapind * 4 + 3;
            if(heightmap[mapind_min]==FLOWMAX)
            {
                heightmapdisp[i*(ynum*2) + j] = uint8_t(0);
                heightmapdisp[i*(ynum*2) + j + ynum] = uint8_t(0);
                heightmapdisp[i*(ynum*2) + j + 2*gridnum] = uint8_t(0);
                heightmapdisp[i*(ynum*2) + j + ynum + 2*gridnum] = uint8_t(0);
            } 
            else
            {
                dispmin = (heightmap[mapind_min]-hmin)*Scale;
                dispmax = (heightmap[mapind_max]-hmin)*Scale;
                dispmean = (heightmap[mapind_mean]-hmin)*Scale;
                dispstd = heightmap[mapind_std] * 1000;
                dispmin = ((dispmin<0?0:dispmin)>255?255:dispmin);
                dispmax = ((dispmax<0?0:dispmax)>255?255:dispmax);
                dispmean = ((dispmean<0?0:dispmean)>255?255:dispmean);
                dispstd = ((dispstd<0?0:dispstd)>255?255:dispstd);
                heightmapdisp[i*(ynum*2) + j] = uint8_t(dispmin);
                heightmapdisp[i*(ynum*2) + j + ynum] = uint8_t(dispmax);
                heightmapdisp[i*(ynum*2) + j + 2*gridnum] = uint8_t(dispmean);
                heightmapdisp[i*(ynum*2) + j + ynum + 2*gridnum] = uint8_t(dispstd);
            }
        }
    }
    cv::Mat heightimg = cv::Mat(xnum*2, ynum*2, CV_8U, heightmapdisp);
    std::string heightmapwin = "Height Map";
    cv::namedWindow(heightmapwin, cv::WINDOW_AUTOSIZE);  
    cv::imshow(heightmapwin, heightimg);
    cv::waitKey(1);
}

void ScrollGrid::inflate_maps(int neighbor_count, bool incremental)
{
    uint8_t *ori_colormap;
    float *ori_heightmap;

    if (incremental==false) // inflate the map based on the colormap and heightmap
    {
        colormap_inflate = new uint8_t[gridnum*3];
        heightmap_inflate = new float[gridnum*4];

        memcpy(colormap_inflate, colormap, gridnum*3*sizeof(uint8_t));
        memcpy(heightmap_inflate, heightmap, gridnum*4*sizeof(float));

        ori_colormap = colormap;
        ori_heightmap = heightmap;
    }
    else // inflate the map based on the already inflated colormap and heightmap
    {
        ori_colormap = new uint8_t[gridnum*3];
        ori_heightmap = new float[gridnum*4];

        memcpy(ori_colormap, colormap_inflate, gridnum*3*sizeof(uint8_t));
        memcpy(ori_heightmap, heightmap_inflate, gridnum*4*sizeof(float));
    }


    int mapind, mapind_color, mapind_height;
    int neighborind_x, neighborind_y, neighborind, neighborind_color, neighborind_height;
    int dispmin, dispmax;
    int neighbor_offsets_x[12] = {-1, -1, -1,  0, 0,  1,  1,  1, 2, -2, 0, 0};
    int neighbor_offsets_y[12] = {-1, 0,  1,  -1, 1, -1,  0, 1, 0, 0, 2, -2};
    int neighbor_color_r_ave, neighbor_color_g_ave, neighbor_color_b_ave;
    float neighbor_height_min_ave, neighbor_height_max_ave,neighbor_height_mean_ave, neighbor_height_std_ave;
    for(int i=0;i<xnum;i++)
    {
        for(int j=0;j<ynum;j++)
        {
            mapind = i*ynum + j;
            mapind_height = mapind * 4;
            mapind_color = mapind * 3;
            if(ori_heightmap[mapind_height]==FLOWMAX) // if current pixel is empty, check its neighbors
            {
                // count how many of its neighbors are not empty
                int valid_neighbor=0;
                neighbor_color_r_ave = 0;
                neighbor_color_g_ave = 0;
                neighbor_color_b_ave = 0;
                neighbor_height_min_ave = 0;
                neighbor_height_max_ave = 0;
                neighbor_height_mean_ave = 0;
                neighbor_height_std_ave = 0;
                for(int k=0;k<12;k++)
                {
                    neighborind_x = i + neighbor_offsets_x[k];
                    neighborind_y = j + neighbor_offsets_y[k];
                    neighborind = neighborind_x*ynum + neighborind_y;
                    neighborind_height = neighborind * 4;
                    neighborind_color = neighborind * 3;
                    if(neighborind_x>=0 && neighborind_x<xnum && 
                        neighborind_y>=0 && neighborind_y<ynum) // neighbor position is valid
                    {
                        if(ori_heightmap[neighborind_height]<FLOWMAX-1) // neighbor is not empty
                        {
                            valid_neighbor ++;
                            neighbor_color_r_ave += ori_colormap[neighborind_color];
                            neighbor_color_g_ave += ori_colormap[neighborind_color+1];
                            neighbor_color_b_ave += ori_colormap[neighborind_color+2];
                            neighbor_height_min_ave += ori_heightmap[neighborind_height];
                            neighbor_height_max_ave += ori_heightmap[neighborind_height+1];
                            neighbor_height_mean_ave += ori_heightmap[neighborind_height+2];
                            neighbor_height_std_ave += ori_heightmap[neighborind_height+3];
                        }
                    }
                }
                if(valid_neighbor>=neighbor_count) // we have enough valid neighbors
                {
                    heightmap_inflate[mapind_height] = neighbor_height_min_ave/valid_neighbor;
                    heightmap_inflate[mapind_height+1] = neighbor_height_max_ave/valid_neighbor;
                    heightmap_inflate[mapind_height+2] = neighbor_height_mean_ave/valid_neighbor;
                    heightmap_inflate[mapind_height+3] = neighbor_height_std_ave/valid_neighbor;
                    colormap_inflate[mapind_color] = uint8_t(neighbor_color_r_ave/valid_neighbor);
                    colormap_inflate[mapind_color+1] = uint8_t(neighbor_color_g_ave/valid_neighbor);
                    colormap_inflate[mapind_color+2] = uint8_t(neighbor_color_b_ave/valid_neighbor);
                }
            } 
        }
    }
}


void ScrollGrid::show_color_map()
{
    cv::Mat colorimg = cv::Mat(xnum, ynum, CV_8UC3, colormap);
    std::string colormapwin = "Color Map";
    cv::namedWindow(colormapwin, cv::WINDOW_AUTOSIZE);  
    cv::imshow(colormapwin, colorimg);
    cv::waitKey(1);
}

// int main(int argc, char** argv)
// {
//     // #include <pcl/pcl_config.h>
//     // std::cout << PCL_VERSION << std::endl;
//     ScrollGrid scrollgrid(0.01, -1.0, 1.0, -2.0, 2.0); 
//     // generate a random point cloud
//     pcl::PointCloud<PointT> pc; 
//     for(int i=0; i<1000000; i++)
//     {
//         // std::uint32_t rgb = ((std::uint32_t)(rand()%256) << 16 | (std::uint32_t)(rand()%256) << 8 | (std::uint32_t)(rand()%256));
//         PointT pt(std::uint8_t(rand()%256),  std::uint8_t(rand()%256),  std::uint8_t(rand()%256));
//         pt.x=(float(rand())/RAND_MAX - 0.5) * 2;
//         pt.y=(float(rand())/RAND_MAX - 0.5) * 4; 
//         pt.z=float(rand())/RAND_MAX,  
//         pc.push_back (pt);
//         // std::cout<<pt<<std::endl;
//     }
//     scrollgrid.pc_to_map(pc);
//     scrollgrid.show_color_map();
//     scrollgrid.show_height_map();

//     pcl::PointCloud<PointT> pc2; 
//     for(int i=0; i<100; i++)
//     {
//         // std::uint32_t rgb = ((std::uint32_t)(rand()%256) << 16 | (std::uint32_t)(rand()%256) << 8 | (std::uint32_t)(rand()%256));
//         PointT pt(std::uint8_t(rand()%256),  std::uint8_t(rand()%256),  std::uint8_t(rand()%256));
//         pt.x=(float(rand())/RAND_MAX - 0.5) * 5;
//         pt.y=(float(rand())/RAND_MAX - 0.5) * 5; 
//         pt.z=float(rand())/RAND_MAX,  
//         pc2.push_back (pt);
//         // std::cout<<pt<<std::endl;
//     }
//     scrollgrid.pc_to_map(pc2);
//     scrollgrid.show_color_map();
//     scrollgrid.show_height_map();
// }