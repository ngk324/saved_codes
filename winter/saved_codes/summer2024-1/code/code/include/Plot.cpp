#include "Plot.hpp"
#include <algorithm>

CvScalar Plot::defaultNodeColor = cvScalar(150, 150, 150);
CvScalar Plot::defaultEdgeColor = cvScalar(150, 150, 150);

int Plot::defaultNodeSize = 1;
double Plot::defaultEdgeThickness = 1;

Plot::Plot(std::string name, int scale, int vPadding, int hPadding, int xMax, int yMax)
    : windowName{name}, plotScale{scale}, verticalPadding{vPadding}, horizontalPadding{hPadding}, maxX{xMax}, maxY{yMax}
{
    defaultNodeSize = plotScale * 0.2;
    defaultEdgeThickness = plotScale * 0.05;
    //blankImg = cv::Mat(cv::Size(maxX + 2 * verticalPadding - 1, maxY + 2 * horizontalPadding - 1), CV_8UC3, cv::Scalar(255, 255, 255));
    blankImg = cv::Mat(maxX, maxY, CV_8UC1, cv::Scalar(0));
    currentImg = blankImg.clone();
    initWindow();
    //displayPlot();
}

cv::Point Plot::transformGraphToPlotCircle(int num) const
{
    double value = ((double)num)*((3.14159265358979323846264338327 * 2.0) / (double)numNodes);
    double x = (cos(value)+1.05) * (double) maxX * 0.475;
    double y = (sin(value)+1.05) * (double) maxY * 0.475;

    return (cvPoint(x,y));
}

void Plot::displayTime(const std::string &time_str)
{
    cv::Size textSize = cv::getTextSize(time_str, cv::FONT_HERSHEY_SIMPLEX, 1, 1, 0);
    cv::Point rightTop(currentImg.size().width - textSize.width - 10, 30);
    rectangle(currentImg, rightTop, cv::Point(textSize.width, -textSize.height) + rightTop,
              cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    putText(currentImg, time_str,
            rightTop, cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 1, cv::LINE_AA);
}
void Plot::displayMethod(const std::string &method_str)
{
    cv::Size textSize = cv::getTextSize(method_str, cv::FONT_HERSHEY_SIMPLEX, 1, 1, 0);
    cv::Point leftTop(10, 30);
    rectangle(currentImg, leftTop, cv::Point(textSize.width, -textSize.height) + leftTop,
              cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
    putText(currentImg, method_str,
            leftTop, cv::FONT_HERSHEY_SIMPLEX, 1, CV_RGB(255, 255, 255), 1, cv::LINE_AA);
}

cv::Point Plot::transformGraphToPlotStateCircle(int num) const
{
    double value = ((double)num)*((3.14159265358979323846264338327 * 2.0) / (double)numNodes);
    double x = (cos(value) + 1.08) * (double) plotScale;
    double y = (sin(value) + 1.12) * (double) plotScale;

    return (cvPoint(x*43,y*43));
}

void Plot::displayState(const Node &n, int num)
{
    double radius = maxX * 0.475;
    cv::Point2d center((maxX / 2.0), (maxX / 2.0));
    double angle_step = 2 * M_PI / numNodes;
    cv::Point plot_pt = center + cv::Point2d(radius * cos(angle_step * num), radius * sin(angle_step * num));
    std::string state_str = std::to_string(n.z).substr(0, 5);
    cv::Size textSize = cv::getTextSize(state_str, cv::FONT_HERSHEY_SIMPLEX, 0.3, 1, 0);
    rectangle(currentImg, plot_pt, cv::Point(textSize.width, -textSize.height) + plot_pt,
              cv::Scalar(255, 255, 255), cv::FILLED, cv::LINE_AA);
    putText(currentImg, state_str,
            plot_pt, cv::FONT_HERSHEY_SIMPLEX, 0.3, CV_RGB(0, 0, 0), 0.5, cv::LINE_AA);
}


void Plot::plotEdgeCircle(double value1, double value2,  double thickness, CvScalar color)
{
    // alpha masking 
   // cv::Mat overlay = currentImg.clone();
   // cv::Mat original = currentImg.clone();
    cv::Mat dummy = currentImg.clone();
    cv::line(currentImg, transformGraphToPlotCircle(value1), transformGraphToPlotCircle(value2), color, thickness);
    //currentImg += dummy;
    //plotEdgeCircle(i,j, 5, edgeColor);
    ///cv::line(currentImg, transformGraphToPlotCircle(value1), transformGraphToPlotCircle(value2), color, thickness);
  //  double alpha = 0.2;
   // cv::addWeighted(overlay, alpha, original, 1 - alpha, 0, currentImg);

}


void Plot::plotNodeCircle(Node &n, double max, double min, int num, int size)
{
    CvScalar color;
    double val;
    if (n.z > 0){
        val = abs(n.z) * (255.0/abs(max));
        color = cvScalar(val);
    }
    else{
        val = abs(n.z) * (255.0/abs(min));
        color = cvScalar(val);
    }
    CvScalar color1 = cvScalar(255);
    size = size*4;
    //cv::circle(currentImg, transformGraphToPlot(n), size, cvScalar(0, 0, 0), -1);
    //cv::circle(currentImg, cvPoint(plotScale * (cos(value) * 0.95), plotScale * (sin(value))*0.95) , size, color, -1);
    double center = (maxX / 2.0);
    double angle_step = 2 * M_PI / numNodes;
    double radius = maxX * .475;
    cv::circle(currentImg, cvPoint(center + radius * cos(angle_step * num), center + radius * sin(angle_step * num)), size, color1, -1);

    //cv::circle(currentImg, cvPoint(plotScale * (cos(value) + verticalPadding), plotScale * (sin(value) + horizontalPadding)) , size, color, -1);
}


void Plot::plotGraphCircle(Graph &g)
{   
    double max = std::max<double>(g.adjacencyMatrix.maxCoeff(),0.0);
    double min = std::min<double>(g.adjacencyMatrix.minCoeff(),0.0);
    
    double minZ = g.nodes[0]->z;
    double maxZ = g.nodes[0]->z;


    for(int i = 0; i < g.nodes.size(); i++){
        if(g.nodes[i]->z > maxZ){
            double maxZ = g.nodes[i]->z;
        }
        if(g.nodes[i]->z < minZ){
            double minZ = g.nodes[i]->z;
        }
    }
   

    double radius = maxX * 0.475;
    cv::Point2d center((maxX / 2.0), (maxX / 2.0));
    double angle_step = 2 * M_PI / numNodes;
    std::vector<cv::Point2d> node_pos;
    for (int i{0}; i < numNodes; i++)
    {
        node_pos.push_back(center + cv::Point2d(radius * cos(angle_step * i), radius * sin(angle_step * i)));
    }
    
    for (int i{0}; i < numNodes - 1; i++)
    {
        for (int j{i+1}; j < numNodes; j++)
        {
            double edgeWeight = g.adjacencyMatrix(i,j);
            if(edgeWeight > 0){
                int color = 255-(((int)(g.adjacencyMatrix.coeff(i,j)*255))/(max-min));
                cv::Mat dummy = blankImg.clone();
                cv::line(dummy, node_pos[i], node_pos[j], cv::Scalar(color));
                currentImg += dummy;
                //plotEdgeCircle(i,j, 5, edgeColor);
            }
        }
    }

    double minVal = 0;
    double maxVal = 0;
    cv::minMaxIdx(currentImg, &minVal, &maxVal);

    currentImg = currentImg*(255.0) / (maxVal-minVal);

    for (int i{0}; i < numNodes; i++)
    {
        plotNodeCircle(*g.nodes[i], maxZ, minZ, i);
    }

    cv::resize(currentImg, currentImg, cv::Size(), plotScale, plotScale);

}


void Plot::initWindow()
{
    cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(windowName, 50, 50);
}


void Plot::displayPlot(bool waitForKey)
{
    cv::imshow(windowName, currentImg);
    cv::imwrite("Optimized_Graph.png",currentImg);
    if (waitForKey){
        char key = (char)cv::waitKey(); // explicit cast
        while (key != 27)
        {
            key = (char)cv::waitKey();
        }
    }
    else
        cv::waitKey(1);
}
