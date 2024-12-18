#ifndef _PLOT_H
#define _PLOT_H

#include "Graph.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


class Plot
{
private:
    std::string windowName;
    cv::Mat blankImg;
    cv::Mat currentImg;

    int plotScale;
    int verticalPadding, horizontalPadding;

    int maxX, maxY;

    static CvScalar defaultNodeColor;
    static CvScalar defaultEdgeColor;

    static int defaultNodeSize;
    static double defaultEdgeThickness;

    cv::Point transformGraphToPlotCircle(int num) const;
    cv::Point transformGraphToPlotStateCircle(int num) const;


public:
    Plot(std::string name, int scale, int vPadding, int hPadding, int xMax, int yMax);

    void displayTime(const std::string &time_str);

    void displayMethod(const std::string &method_str);

    void displayState(const Node &n, int num);

    void plotEigen(const Graph &g, Eigen::VectorXf &vec, int size = defaultNodeSize);

    void plotNodeCircle(Node &n, double max, double min, int num, int size = 5);

    void plotEdgeCircle(double value1, double value2, double thickness=defaultEdgeThickness, CvScalar color=defaultEdgeColor);
        
    void plotGraphCircle(Graph &g);

    void initWindow();

    void displayPlot(bool waitForKey = false);

    double round(double var, int place);
};

#endif
