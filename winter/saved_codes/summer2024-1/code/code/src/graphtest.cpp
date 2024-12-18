#include "include/gdMain.hpp"

#include "../include/Graph.hpp"



// My headers
#include "include/Plot.hpp"
#include "include/Dynamics.hpp"
#include "include/Force.hpp"



#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <string>

// global variables
int idx;    // variable for node where force is applied
bool beforeGrad;    // variable to tell whether simulation is running on plot before or after gradient descent
int simNum; // simulation number
double randProb;    // probability to use for sampling from cauchy
double h;   // h parameter for cauchy
double frequencyFromUniform; // freq chosen from uniform (eigenvalue)
int numNodes;   // number of nodes 



// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

void render_display(cv::Mat &img, int scale = 1)
{
    cv::Mat display_mat = img.clone();
    cv::resize(display_mat, display_mat, cv::Size(), scale, scale);
    cv::namedWindow("Display", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display", display_mat);
    char key = (char)cv::waitKey();
    while (key != 27)
    {
        key = (char)cv::waitKey();
    }
}

int main(int argc, char **argv)
{
    int numNodes = 96;

    double max = 0;
    double min = 0.1;

     std::vector<double> buffer;
    std::string line;
    std::ifstream myFile("LapResults.txt");
    Eigen::MatrixXf A(numNodes, numNodes);
    Eigen::MatrixXf C(numNodes, numNodes);
    Eigen::MatrixXf D(numNodes, numNodes);
    Eigen::MatrixXf L(numNodes, numNodes);
    while(getline(myFile, line))
    {
        std::istringstream lineStream(line);
        double first;
        lineStream >> first;
        buffer.push_back(first);
    }
    int counter = 0;
    for (int i{0}; i < numNodes; i++)
    {
        for (int j{0}; j < numNodes; j++)
        {
            C(i,j) = 0;
            if(i == j){
                D(i,j) = buffer[counter];
                A(i,j) = 0;
            }
            else{
                A(i,j) = -buffer[counter];
                A(j,i) = -buffer[counter];
                if(max < -buffer[counter]){
                    max = -buffer[counter];
                }
                D(i,j) = 0;
                D(j,i) = 0;
                if(buffer[counter] < 0){
                    C(i,j) = 1;
                }
            }
            counter++;
        }
    }
    L = D - A;




    int size = 1000; // Size of the initial matrix (this could be as large as the screen resolution,
                   // picking a larger size might help us avoid undesired overlaps between the edges,
                   // stemming from the same node)

    int nNodes = 96;

    cv::Mat init_img = cv::Mat(size, size, CV_8UC1, cv::Scalar(0)); // A matrix storing single scalar per cell (initially 0)


    std::vector<cv::Point2d> node_pos;
    double radius = size * 0.5;
    cv::Point2d center(size / 2.0, size / 2.0);
    double angle_step = 2 * M_PI / nNodes;
    // Distribute the nodes around the circle
    for (int i{0}; i < nNodes; i++)
    {
        node_pos.push_back(center + cv::Point2d(radius * cos(angle_step * i), radius * sin(angle_step * i)));
    }

    cv::Mat line_mat = init_img.clone();

    // Plotting the lines on the matrix
    for (int i{0}; i < nNodes; i++)
    {
        for (int j{i + 1}; j < nNodes; j++)
        {
            double edgeWeight = A(i,j);
            int color = 255-(((int)(A(i,j)*255))/(max-min));
            double weight = (j - i) * 2;          // Some weight calculated here to test whether pixel values are summed correctly
            cv::Mat dummy_mat = init_img.clone(); // a dummy mat to store a single line;
            cv::line(dummy_mat, node_pos[i], node_pos[j], cv::Scalar(color));
            line_mat += dummy_mat; // this should add the line on top of the existing matrix

        }
    }


    double minVal{0};
    double maxVal{0};
    cv::minMaxIdx(line_mat, &minVal, &maxVal);

    // Scaling the matrix elements to range 0-255 in the end
    cv::Mat scaled_line_mat = line_mat * (255.0) / maxVal;
    render_display(scaled_line_mat, 1000.0 / size);
}