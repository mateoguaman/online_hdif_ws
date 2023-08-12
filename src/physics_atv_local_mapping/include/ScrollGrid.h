#include <pcl/point_types.h>
#include <pcl/features/feature.h>
// #include <pcl/ros/conversions.h>

#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <iostream>

typedef pcl::PointXYZRGB  PointT;


class ScrollGrid {
public:
    ScrollGrid();

    // ScrollGrid(double mapresolution, 
    //            double x_min, double x_max, 
    //            double y_min, double y_max);
    void init(double mapresolution, 
               double x_min, double x_max, 
               double y_min, double y_max);

    ScrollGrid& operator=(const ScrollGrid &other);
    ~ScrollGrid();
    void pc_to_map(const pcl::PointCloud<PointT> &pc); 
    float* getHeightmap();
    uint8_t* getColormap();
    void show_height_map();
    void show_color_map();
    int getMapHeight();
    int getMapWidth();
    void inflate_maps(int neighbor_count, bool incremental);
    float* getInflateHeightmap();
    uint8_t* getInflateColormap();

private:
    void initialize_elevation_map();
    void reset_elevation_map();

    // void pc_xy_to_grid_ind(const double x, const double y, int &xind, int &yind);

    double resolution, xmin, xmax, ymin, ymax;
    int xnum, ynum, gridnum;
    float *heightmap, *heightmap_inflate; 
    uint8_t *colormap, *colormap_inflate;
    int *point_count; // used to calculate height mean and std
    float *height_mean_sq; // used to calculate height mean and std
};