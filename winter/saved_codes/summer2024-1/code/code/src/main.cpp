// OpenCV libraries
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <math.h>
#include <numeric>
#include <iostream>

// My headers
#include "include/Plot.hpp"
#include "include/Dynamics.hpp"
#include "include/Force.hpp"

#include <iostream>
#include <vector>

void plot1DPointsStraight(const std::vector<double>& values, int point1, int point2, int width = 800*1.5, int height = 600*1.5) {
    if (values.empty()) {
        std::cout << "Error: The input vector is empty." << std::endl;
        return;
    }

    // Find the mini and max values 
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());

    cv::Mat plot_image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double scale_x = static_cast<double>(width/max_val);

    // Plot the points
    for (size_t i = 0; i < values.size(); ++i) {
        int x = static_cast<int>(values[i] * scale_x);
        int y = static_cast<int>(height / 2);
        if(i == point1){
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        }
        else if(i == point2){
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        }
        else{
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 0), -1);
        }
    }

    cv::imshow("1D Plot", plot_image);
    cv::waitKey(0);
}

void plot1DPointsCurved(const std::vector<double>& values, int point1, int point2, int width = 800*1.5, int height = 600*1.5) {
    if (values.empty()) {
        std::cout << "Error: The input vector is empty." << std::endl;
        return;
    }

    // Find the minimum and maximum values
    double min_val = *std::min_element(values.begin(), values.end());
    double max_val = *std::max_element(values.begin(), values.end());
    cv::Mat plot_image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double scale_x = static_cast<double>(width) / (values.size() - 1);
    double scale_y = static_cast<double>(height) / (max_val - min_val);

    // Plot the points
    for (size_t i = 0; i < values.size(); ++i) {
        int x = static_cast<int>(i * scale_x);
        int y = static_cast<int>((max_val - values[i]) * scale_y);
        if(i == point1){
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        }
        else if(i == point2){
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        }
        else{
            cv::circle(plot_image, cv::Point(x, y), 3, cv::Scalar(0, 0, 0), -1);
        }
    }
    cv::imshow("1D Plot", plot_image);
    cv::waitKey(0);
}

/*
// max
double compute_var_grad(const std::vector<double>& values, Eigen::Matrix<float, -1, 1> vki, Eigen::Matrix<float, -1, 1> vkj, double stdev){
    double d_var = 0; 

    double h = stdev;

    double printVal;

    // loops through each element of lambda_j
    for(int i = 1; i < values.size(); i++){
        int index = i;
        double maxVal;
        int maxIndex;

        // finds lambda_k that maximizes 
        if(index != 1){
            //maxVal = ((4*h*h)/values[1]) * (1/(pow(pow(sqrt(values[1])-sqrt(values[index]),2) + h * h, 2)));   
            maxVal = ((4*h*h)/(values[index])) * (1/(pow(pow(sqrt(values[index])-sqrt(values[1]),2) + h * h, 2)));
            maxIndex = 1;
            //std::cout << " \nTest i,j " << i << " , " << 0 << " " << (a / (2 * values[1])) * exp(-b * (pow(values[1] - values[index],2)));
            
        }
        else{
            //maxVal = ((4*h*h)/values[2]) * (1/(pow(pow(sqrt(values[2])-sqrt(values[index]),2) + h * h, 2)));   
            maxVal = ((4*h*h)/(values[index])) * (1/(pow(pow(sqrt(values[index])-sqrt(values[2]),2) + h * h, 2)));
            maxIndex = 2;
            //std::cout << " \nTest i,j " << i << " , " << 0 << " " << (a / (2 * values[1])) * exp(-b * (pow(values[1] - values[index],2)));
            
        }
        for(int j = 0; j < values.size(); j++){
            if(j != index && j != maxIndex){
                //double testVal = ((4*h*h)/values[j]) * (1/(pow(pow(sqrt(values[j])-sqrt(values[index]),2) + h * h, 2))); 
                double testVal = ((4*h*h)/(values[index])) * (1/(pow(pow(sqrt(values[index])-sqrt(values[j]),2) + h * h, 2)));
                if(testVal > maxVal){
                    maxVal = testVal;
                    maxIndex = j;
                }
            }
            //std::cout << " \nTest i,j " << i << " , " << j << " " << (a / (2 * values[j])) * exp(-b * (pow(values[j] - values[index],2)));
        }
        printVal = maxVal;
        //std::cout << "Val " << maxVal << " Index " << maxIndex << "\n ";

        double min_func_val = -((4 * h*h)/(sqrt(values[index])*pow(values[index],1.5))) * ((3 * values[index]- 4 * sqrt(values[maxIndex])*sqrt(values[index])+values[maxIndex]+h*h)/(pow(h*h+pow(sqrt(values[index])-sqrt(values[maxIndex]),2),3)));
        d_var = d_var + min_func_val * (vki[i] * vki[i] + vkj[i]*vkj[i] - 2.0*vki[i]*vkj[i]);

    }
    //std::cout << "\nObj func: " << d_var;
    return d_var;
}
*/

/*
// max with + k
double compute_var_grad(const std::vector<double>& values, Eigen::Matrix<float, -1, 1> vki, Eigen::Matrix<float, -1, 1> vkj, double stdev){
    double d_var = 0; 

    double h = stdev;

    double printVal;
    
    // loops through each element of lambda_j
    for(int i = 0; i < values.size(); i++){
        int index = i;
        double maxVal;
        int maxIndex;
        double k = .010;

        //h = h * values[index];

        // finds lambda_k that maximizes 
        if(index != 0){ 
            maxVal = ((4*h*h)/(values[index]+k)) * (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[0]+k),2) + h * h, 2)));
            //maxVal = (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[0]+k),2) + h * h, 2)));
            maxIndex = 0;
            //std::cout << " \nTest i,j " << i << " , " << 0 << " " << (a / (2 * values[1])) * exp(-b * (pow(values[1] - values[index],2)));
            
        }
        else{
            maxVal = ((4*h*h)/(values[index]+k)) * (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[1]+k),2) + h * h, 2)));
            //maxVal = (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[1]+k),2) + h * h, 2)));
            maxIndex = 1;
            //std::cout << " \nTest i,j " << i << " , " << 0 << " " << (a / (2 * values[1])) * exp(-b * (pow(values[1] - values[index],2)));
            
        }
        for(int j = 0; j < values.size(); j++){
            if(j != index && j != maxIndex){
                double testVal = ((4*h*h)/(values[index]+k)) * (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[j]+k),2) + h * h, 2)));
                //double testVal = (1/(pow(pow(sqrt(values[index]+k)-sqrt(values[j]+k),2) + h * h, 2)));
                if(testVal > maxVal){
                    maxVal = testVal;
                    maxIndex = j;
                }
            }
            //std::cout << " \nTest i,j " << i << " , " << j << " " << (a / (2 * values[j])) * exp(-b * (pow(values[j] - values[index],2)));
        }
        printVal = maxVal;
        //std::cout << "Val " << maxVal << " Index " << maxIndex << "\n ";

        double min_func_val = -((4 * h*h)/(sqrt(values[index] + k)*pow(values[index] + k,1.5))) * ((3 * values[index] + k- 4 * sqrt(values[maxIndex]+k)*sqrt(values[index]+k)+values[maxIndex]+k+h*h)/(pow(h*h+pow(sqrt(values[index]+k)-sqrt(values[maxIndex]+k),2),3)));
        //double min_func_val = -(1/sqrt(values[index] + k)) * ((2 * (sqrt(values[index]+k)-sqrt(values[maxIndex]+k)))/(pow(h*h+pow(sqrt(values[index]+k)-sqrt(values[maxIndex]+k),2),3)));
        d_var = d_var + min_func_val * (vki[i] * vki[i] + vkj[i]*vkj[i] - 2.0*vki[i]*vkj[i]);

    }
    //std::cout << "\nObj func: " << d_var;
    return d_var;
}
*/


// double max
double compute_var_grad(const std::vector<double>& values, Eigen::Matrix<float, -1, 1> vki, Eigen::Matrix<float, -1, 1> vkj, double stdev, bool print){
    double d_var = 0; 

    double h = stdev;

    double maxVal;

    // loops through each element of lambda_j
    for(int i = 0; i < values.size(); i++){
        int maxIndex;
        if(i != 1){
            maxVal = ((16 * h * h) * (sqrt(values[1])-sqrt(values[i]))) / ((pow(pow(sqrt(values[1])-sqrt(values[i]), 2)+h*h,3)));
            maxIndex = 1;
        }
        else{
            maxVal = ((16 * h * h) * (sqrt(values[0])-sqrt(values[i]))) / ((pow(pow(sqrt(values[0])-sqrt(values[i]), 2)+h*h,3)));
            maxIndex = 0;
        }        
        for(int l = 0; l < values.size(); l++){
            if(i != l){
                if(abs(values[l]-values[i]) > 0.0002){
                    double testVal = ((16 * h * h) * (sqrt(values[l])-sqrt(values[i]))) / ((pow(pow(sqrt(values[l])-sqrt(values[i]), 2)+h*h,3)));
                    if(testVal > maxVal){
                        maxIndex = l;
                        maxVal = testVal;
                    }
                }
                else{
                    double testVal = ((16 * h * h) * (sqrt(values[l]+.1)-sqrt(values[i]))) / ((pow(pow(sqrt(values[l])-sqrt(values[i]), 2)+h*h,3)));
                    if(testVal > maxVal){
                        maxIndex = l;
                        maxVal = testVal;
                    }
                }
            }
        }
        d_var = d_var + maxVal * (vki[i] * vki[i] + vkj[i]*vkj[i]  - 2.0*vki[i]*vkj[i]);
    }
    //std::cout << "\n" << d_var;
    return d_var;
}


/*
// double max
double compute_var_grad(const std::vector<double>& values, Eigen::Matrix<float, -1, 1> vki, Eigen::Matrix<float, -1, 1> vkj, double stdev, bool print){
    double d_var = 0; 

    double h = stdev;

    double printVal;
    
    int maxIndexK = 0;
    int maxIndexJ = 1;
    double maxVal = ((4*h*h)) * (1/(pow(pow(sqrt(values[maxIndexJ])-sqrt(values[maxIndexK]),2) + h * h, 2)));
    double k = 0.0;
    // loops through each element of lambda_j
    for(int i = 0; i < values.size(); i++){

        //h = h * values[index];

        // finds lambda_k that maximizes 
        for(int j = 0; j < values.size(); j++){
            if(j != i){
                double testVal = ((4*h*h)) * (1/(pow(pow(sqrt(values[i])-sqrt(values[j]),2) + h * h, 2)));
                if(testVal > maxVal){
                    maxVal = testVal;
                    maxIndexK = j;
                    maxIndexJ = i;
                }
            }
            //std::cout << " \nTest i,j " << i << " , " << j << " " << (a / (2 * values[j])) * exp(-b * (pow(values[j] - values[index],2)));
        }
        printVal = maxVal;
        //std::cout << "Val " << maxVal << " Index " << maxIndex << "\n ";
    }
    //double min_func_val = -((4 * h*h)/(sqrt(values[maxIndexJ] + k)*pow(values[maxIndexJ] + k,1.5))) * ((3 * values[maxIndexJ] + k- 4 * sqrt(values[maxIndexK]+k)*sqrt(values[maxIndexJ]+k)+values[maxIndexK]+k+h*h)/(pow(h*h+pow(sqrt(values[maxIndexJ]+k)-sqrt(values[maxIndexK]+k),2),3)));
    //double min_func_val = -(1/sqrt(values[maxIndexJ] + k)) * ((2 * (sqrt(values[maxIndexJ]+k)-sqrt(values[maxIndexK]+k)))/(pow(h*h+pow(sqrt(values[maxIndexJ]+k)-sqrt(values[maxIndexK]+k),2),3)));
    double min_func_val_j, min_func_val_k;
    if((abs(values[maxIndexJ]-values[maxIndexK])) > 0.0002){
        min_func_val_j = -((8*h*h*(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ])))/(sqrt(values[maxIndexJ])*pow(h*h+pow(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ]),2),3)));
        min_func_val_k = ((8*h*h*(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ])))/(sqrt(values[maxIndexK])*pow(h*h+pow(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ]),2),3)));
        //min_func_val = ((4 * h*h)/(sqrt(values[maxIndexJ])*pow(values[maxIndexJ],1.5))) * ((-3 * values[maxIndexJ] + 4 * sqrt(values[maxIndexK])*sqrt(values[maxIndexJ])-values[maxIndexK]-h*h)/(pow(h*h-pow(sqrt(values[maxIndexJ])+sqrt(values[maxIndexK]),2),3)));

    }
    else{
        min_func_val_j = -((8*h*h*(-sqrt(values[maxIndexK]+.1)+sqrt(values[maxIndexJ])))/(sqrt(values[maxIndexJ])*pow(h*h+pow(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ]),2),3)));
        min_func_val_k = ((8*h*h*(-sqrt(values[maxIndexK]+.1)+sqrt(values[maxIndexJ])))/(sqrt(values[maxIndexK])*pow(h*h+pow(-sqrt(values[maxIndexK])+sqrt(values[maxIndexJ]),2),3)));
        //min_func_val = ((4 * h*h)/(sqrt(values[maxIndexJ])*pow(values[maxIndexJ],1.5))) * ((-3 * values[maxIndexJ] + 4 * sqrt(values[maxIndexK])*sqrt(values[maxIndexJ])-values[maxIndexK]-h*h)/(pow(h*h-pow(sqrt(values[maxIndexJ])+sqrt(values[maxIndexK]),2),3)));

    }
    //d_var = (min_func_val_j * (vki[maxIndexJ] * vki[maxIndexJ] + vkj[maxIndexJ]*vkj[maxIndexJ] - 2.0*vki[maxIndexJ]*vkj[maxIndexJ])); - (min_func_val_k * (vki[maxIndexK] * vki[maxIndexK] + vkj[maxIndexK]*vkj[maxIndexK] - 2.0*vki[maxIndexK]*vkj[maxIndexK]));
        d_var = (maxVal * (vki[maxIndexJ] * vki[maxIndexJ] + vkj[maxIndexJ]*vkj[maxIndexJ] - 2.0*vki[maxIndexJ]*vkj[maxIndexJ])) - (maxVal * (vki[maxIndexK] * vki[maxIndexK] + vkj[maxIndexK]*vkj[maxIndexK] - 2.0*vki[maxIndexK]*vkj[maxIndexK]));
    //std::cout << "\nMax Indices " << maxIndexJ << " " << maxIndexK << " " << values[maxIndexJ] << " " << values[maxIndexK] << " " << (1/(pow(pow(sqrt(values[maxIndexJ])-sqrt(values[maxIndexK]),2) + h * h, 2))) << " " << maxVal << " " << d_var;
    if(print == true){
        std::cout << "\nMax Indices " << maxIndexJ << " " << maxIndexK << " " << values[maxIndexJ] << " " << values[maxIndexK];    
        std::cout << "\nObj func: " << maxVal;    
        //plot1DPointsCurved(values, maxIndexJ, maxIndexK);
    }
    return d_var;
}
*/

// rounds to 4 decimal places
double round(double var, int place)
{
    double value = (int)(var * 10000 +.5);
    return (double)value / 10000;
}

void plotHeatMap(Eigen::Matrix<float, -1, 1> arr, int size, char* name)
{
    int width = 400;
    int height = 400;

    double heatmap[width][height];

    // Populate the heatmap array with value
    int index = 0;
    for(int k = 0; k < size * size; k++){
        for (int i = 0; i < width / size; i++){
            for (int j = 0; j < height / size; j++){
                // Calculate value based on position (example calculation shown)
                heatmap[(k / size) * (width / size) + i][(k%size) * (width / size) + j] = arr[index];
            }
        }
        index++;
    }
    // Finds the min and max values in the heatmap array
    double minValue = heatmap[0][0];
    double maxValue = heatmap[0][0];
    //double minValue = -1;
    //double maxValue = 1;
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            if (heatmap[i][j] < minValue)
                minValue = heatmap[i][j];
            if (heatmap[i][j] > maxValue)
                maxValue = heatmap[i][j];
        }
    }

    // Create a Mat object to store the heatmap image
    cv::Mat image(height, width, CV_8UC3);

    // Iterate through each pixel and set the color based on the corresponding value in the array
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            // Normalize the value to the range [0, 255]
            int intensity = static_cast<int>((heatmap[i][j] - minValue) / (maxValue - minValue) * 255);

            // Set the color of the current pixel based on the intensity
            cv::Vec3b color(intensity, intensity, intensity);

            // Set the color of the current pixel
            image.at<cv::Vec3b>(j, i) = color;
        }
    }

    cv::imshow(name, image);

    cv::waitKey(0);

    //cv::destroyAllWindows();
}

void plotHistogram(const std::vector<double>& values, int maxFrequency, int itNum)
{
    double bins = 20.0;
    // Find the minimum and maximum values in the vector
    double minValue = 0;
    double maxValue = 8;
    //double minValue = *std::min_element(values.begin(), values.end());
    //double maxValue = *std::max_element(values.begin(), values.end());

    // Calculate the histogram
    std::vector<int> histogram(bins, 0);
    float binSize = static_cast<float>(maxValue - minValue + .1) / bins;

    for (double value : values) {
        int binIndex = static_cast<int>((value - minValue) / binSize);
        histogram[binIndex]++;
    }

    // Find the maximum frequency in the histogram
    //int maxFrequency = *std::max_element(histogram.begin(), histogram.end());

    // Create a histogram visualization image
    int histWidth = 512*2, histHeight = 400*2;
    int histWidth2 = histWidth*1.25,histHeight2 = histHeight *1.25;
    cv::Mat histImage(histHeight2, histWidth2, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the histogram bars
    int binWidth = cvRound(static_cast<double>(histWidth) / bins);
    for (int i = 0; i < bins; i++) {
        int barHeight = cvRound(static_cast<double>(histogram[i]) / maxFrequency * histHeight);
        cv::rectangle(histImage, cv::Point(i * binWidth + ((histWidth2-histWidth)/2), histHeight + ((histHeight2-histHeight)/2)), cv::Point((i + 1) * binWidth + ((histWidth2-histWidth)/2), histHeight + ((histHeight2-histHeight)/2) - barHeight), cv::Scalar(0, 0, 0), -1);

        // Add bin values to the x-axis of the graph
        std::ostringstream binValue;
        binValue << std::fixed << std::setprecision(0) << minValue + binSize * i;
        cv::putText(histImage, binValue.str(), cv::Point(i * binWidth + ((histWidth2-histWidth)/2), histHeight2 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // Add a scale to the y-axis to show bin height
    for (int i = 0; i <= 10; i += 2) {
        int y = histHeight2 - cvRound(static_cast<double>(i) / 10 * histHeight) - ((histHeight2-histHeight)/2);
        cv::putText(histImage, std::to_string(i * maxFrequency / 10), cv::Point(5, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        // Draw the dotted lines
        cv::Point pt1(30, y);
        cv::Point pt2(histWidth2 - 30, y);
        cv::line(histImage, pt1, pt2, cv::Scalar(200, 200, 200), 1, cv::LINE_4);
    }

    // Create a window and display the histogram
    cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
    cv::imshow("Histogram", histImage);
    std::__cxx11::basic_string<char> name = "tests/Iteration_scaled2"+std::to_string(itNum) + ".jpg";
    cv::imwrite(name, histImage);  
    cv::waitKey(0);
    cv::destroyAllWindows();
}

int plotHistogramFirst(const std::vector<double>& values, int itNum)
{
    double bins = 20;
    // Find the minimum and maximum values in the vector
    double minValue = 0;
    double maxValue = 8;
    //double minValue = *std::min_element(values.begin(), values.end());
    //double maxValue = *std::max_element(values.begin(), values.end());

    // Calculate the histogram
    std::vector<int> histogram(bins, 0);
    float binSize = static_cast<float>(maxValue - minValue + .1) / bins;

    for (double value : values) {
        int binIndex = static_cast<int>((value - minValue) / binSize);
        histogram[binIndex]++;
    }

    // Find the maximum frequency in the histogram
    int maxFrequency = *std::max_element(histogram.begin(), histogram.end());

    // Create a histogram visualization image
    int histWidth = 512*2, histHeight = 400*2;
    int histWidth2 = histWidth*1.25,histHeight2 = histHeight *1.25;
    cv::Mat histImage(histHeight2, histWidth2, CV_8UC3, cv::Scalar(255, 255, 255));

    // Draw the histogram bars
    int binWidth = cvRound(static_cast<double>(histWidth) / bins);
    for (int i = 0; i < bins; i++) {
        int barHeight = cvRound(static_cast<double>(histogram[i]) / maxFrequency * histHeight);
        cv::rectangle(histImage, cv::Point(i * binWidth + ((histWidth2-histWidth)/2), histHeight + ((histHeight2-histHeight)/2)), cv::Point((i + 1) * binWidth + ((histWidth2-histWidth)/2), histHeight + ((histHeight2-histHeight)/2) - barHeight), cv::Scalar(0, 0, 0), -1);

        // Add bin values to the x-axis of the graph
        std::ostringstream binValue;
        binValue << std::fixed << std::setprecision(0) << minValue + binSize * i;
        cv::putText(histImage, binValue.str(), cv::Point(i * binWidth + ((histWidth2-histWidth)/2), histHeight2 - 10), cv::FONT_HERSHEY_SIMPLEX, 0.2, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    // Add a scale to the y-axis to show bin height
    for (int i = 0; i <= 10; i += 2) {
        int y = histHeight2 - cvRound(static_cast<double>(i) / 10 * histHeight) - ((histHeight2-histHeight)/2);
        cv::putText(histImage, std::to_string(i * maxFrequency / 10), cv::Point(5, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
        // Draw the dotted lines
        cv::Point pt1(30, y);
        cv::Point pt2(histWidth2 - 30, y);
        cv::line(histImage, pt1, pt2, cv::Scalar(200, 200, 200), 1, cv::LINE_4);
    }

    // Create a window and display the histogram
    cv::namedWindow("Histogram", cv::WINDOW_NORMAL);
    cv::imshow("Histogram", histImage);
    std::__cxx11::basic_string<char> name = "tests/Iteration_scaled2"+std::to_string(itNum) + ".jpg";
    cv::imwrite(name, histImage);  
    cv::waitKey(0);
    cv::destroyAllWindows();
    return maxFrequency;
}



// get eigen pair with first float being eigenvalue and second Eigen::VectorXf being eigenvectors
std::vector<std::pair<float, Eigen::VectorXf>> get_eigen_pairs(Eigen::MatrixXf &matrix)
{
    std::vector<std::pair<float, Eigen::VectorXf>> eigen_pairs;
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
    Eigen::EigenSolver<Eigen::MatrixXf> solver(matrix);
    Eigen::VectorXcf eigVals = solver.eigenvalues();
    Eigen::MatrixXcf eigVecs = solver.eigenvectors();

    for (int i{0}; i < eigVals.size(); i++)
    {
        /*if (eigVals(i).imag() != 0)
        {
            std::cout << "Complex eigenvalue!" << std::endl;
            exit(EXIT_FAILURE);
        }*/
        eigen_pairs.push_back(std::make_pair(eigVals(i).real(), eigVecs.col(i).real()));
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), [&](const std::pair<float, Eigen::VectorXf> &a, const std::pair<float, Eigen::VectorXf> &b) -> bool
              { return a.first < b.first; });
    return eigen_pairs;
}

// get eigen pair with first float being eigenvalue and second Eigen::VectorXf being eigenspace matrices transposed
std::vector<std::pair<float, Eigen::VectorXf>> get_eigen_pairs_transpose(Eigen::MatrixXf &matrix)
{
    std::vector<std::pair<float, Eigen::VectorXf>> eigen_pairs;
    //Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(matrix);
    Eigen::EigenSolver<Eigen::MatrixXf> solver(matrix);
    Eigen::VectorXcf eigVals = solver.eigenvalues();
    Eigen::MatrixXcf eigVecs = solver.eigenvectors();

    for (int i{0}; i < eigVals.size(); i++)
    {
        /*if (eigVals(i).imag() != 0)
        {
            std::cout << "Complex eigenvalue!" << std::endl;
            exit(EXIT_FAILURE);
        }*/
        eigen_pairs.push_back(std::make_pair(eigVals(i).real(), eigVecs.row(i).real()));
    }
    std::sort(eigen_pairs.begin(), eigen_pairs.end(), [&](const std::pair<float, Eigen::VectorXf> &a, const std::pair<float, Eigen::VectorXf> &b) -> bool
              { return a.first < b.first; });
    return eigen_pairs;
}

int findModeIndex(const std::vector<double>& nums) {
    std::unordered_map<float, float> freq;
    int maxFreq = 0;
    int modeIndex = -1;

    // Count the frequency of each element
    for (int i = 0; i < nums.size(); i++) {
        freq[nums[i]]++;
        if (freq[nums[i]] > maxFreq) {
            maxFreq = freq[nums[i]];
            modeIndex = i;
        }
    }
    std::cout << "Mode FreqCount: " << maxFreq;
    if (modeIndex != -1) {
        std::cout << "\nMode Index: " << modeIndex << " Mode Value: " << nums[modeIndex] << std::endl;
    } else {
        std::cout << "\nNo mode found." << std::endl;
    }
    return modeIndex;
}

Graph scale(Graph my_graph, double scale_factor, double trace){
    int counter = 0;
    for(int i = 0; i < my_graph.nodes.size(); i++){
        for(int j = 0; j < (my_graph.nodes[i])->neighbors.size(); j++){
            if((*my_graph.nodes[i]).id < (*my_graph.nodes[i]->neighbors[j]).id){                              
                my_graph.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) * (scale_factor/trace);
                my_graph.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) * (scale_factor/trace);
                counter++;
            }
        }
    }
}

std::vector<double> vector_projection(std::vector<double> var_gradient){
    std::vector<double> dot_vec;
    double vec_dot_unit_norm = 0;
    for(int i = 0; i < var_gradient.size(); i++){
        //std::cout << "\n Index " << i << " grad " << var_gradient[i] << "\n";
        vec_dot_unit_norm = vec_dot_unit_norm + (var_gradient[i] * (1.0/sqrt(var_gradient.size())));
    }

    for(int i = 0; i < var_gradient.size(); i++){
        dot_vec.push_back(var_gradient[i] - (vec_dot_unit_norm * (1.0/sqrt(var_gradient.size()))));
    }
    return dot_vec;
}

double calc_stdev(double mean, std::vector<double> ev){
    std::vector<double> diff(ev.size());
    std::transform(ev.begin(), ev.end(), diff.begin(),
    std::bind2nd(std::minus<double>(), mean));
    double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / ev.size());
    return stdev;
}

void print_vec(std::vector<double> ev){
    std::cout << "\n";
    for(int i = 0; i < ev.size(); i++){
        std::cout << ev[i] << ", ";
    }
}

void calc_obj_function(std::vector<double> values, double stdev){
    double d_var = 0; 

    double h = stdev;

    double printVal;
    
    int maxIndexK = 0;
    int maxIndexJ = 1;
    double maxVal = ((4*h*h)) * (1/(pow(pow(sqrt(values[maxIndexJ])-sqrt(values[maxIndexK]),2) + h * h, 2)));
    double k = 0.0;
    // loops through each element of lambda_j
    for(int i = 0; i < values.size(); i++){
        // finds lambda_k that maximizes 
        for(int j = 0; j < values.size(); j++){
            if(j != i){
                double testVal = ((4*h*h)) * (1/(pow(pow(sqrt(values[i])-sqrt(values[j]),2) + h * h, 2)));
                if(testVal > maxVal){
                    maxVal = testVal;
                    maxIndexK = j;
                    maxIndexJ = i;
                }
            }
            //std::cout << " \nTest i,j " << i << " , " << j << " " << (a / (2 * values[j])) * exp(-b * (pow(values[j] - values[index],2)));
        }
        printVal = maxVal;
        //std::cout << "Val " << maxVal << " Index " << maxIndex << "\n ";
    }
    std::cout << "\nObj Func: " << printVal;
}

std::vector<int> create_neg_val_index_locations(Graph my_graph){
    std::vector<int> neg_index;
    for(int i = 0; i < my_graph.nodes.size(); i++){
        for(int j = 0; j < (my_graph.nodes[i])->neighbors.size(); j++){
            neg_index.push_back(-1);
        }    
    }
    return neg_index;
}    

std::vector<double> create_eigenvalue_vector(auto eigen_pairs, bool printVals){
    std::vector<double> ev;
    for (int i{0}; i < eigen_pairs.size(); i++)
    {
        if(i == 0){
            std::cout << "\nMin EV: " << round(eigen_pairs[i].first,4);
        }
        else if(i == eigen_pairs.size() - 1){
            std::cout << "\nMax EV: " << round(eigen_pairs[i].first,4);
        }
        ev.push_back(round(eigen_pairs[i].first,4));
        if(printVals == true){
            std::cout << ev[i] << ", ";
        }
    }
    return ev;
}

double calc_trace(Graph my_graph, int MAX_X){
    double trace = 0;
    for(int i = 0; i < MAX_X*MAX_X; i++){
        trace = trace + my_graph.laplacianMatrix(i,i);
    }
    return trace;
}

double calc_adaGrad(std::vector<double> dot_vec){
    double AdaGrad_ep;
    for(int x = 0; x < dot_vec.size(); x++){
        AdaGrad_ep = AdaGrad_ep + dot_vec[x]*dot_vec[x];
    }
    return AdaGrad_ep;
}

int main(int argc, char *argv[])
{
    bool decentralizedAlg = false;
    if (argc == 2)
    {
        decentralizedAlg = std::stoi(argv[1]);
    }
    // Initialize and construct simple graph
    int xGrid{10}, yGrid{10};
    Graph my_graph;
    my_graph.constructSimpleGraph(xGrid, yGrid);
    my_graph.computeMatrices();

    int MAX_X = 10;
    int MAX_Y = 10;
    int PLOT_SCALE = 40;
    int vPad = 2;
    int hPad = 2;

/*
    double damping{0.1}, stiffness{5}, epsilon{0.01};
    double amp{1};
*/

    //Eigen::MatrixXf B_Matrix = stiffness * (my_graph.laplacianMatrix + epsilon * Eigen::MatrixXf::Identity(my_graph.nodes.size(), my_graph.nodes.size()));
    Eigen::MatrixXf B_Matrix = my_graph.laplacianMatrix;
    auto eigen_pairs = get_eigen_pairs(B_Matrix);
    float chosenEigVal{0};
    Eigen::VectorXf chosenEigVec;

    // creates single vector of eigenvalues
    std::vector<double> ev = create_eigenvalue_vector(eigen_pairs, true);
 
    // finds mode
    int modeIndex = findModeIndex(ev);

    // calculates variance
    double mean = std::accumulate(ev.begin(), ev.end(), 0.0) / ev.size();
    double stdev = calc_stdev(mean, ev);
    std::cout << "\n" << "Variance : " << stdev * stdev << "\n";

/*
    // Generate plot
    chosenEigVal = eigen_pairs[modeIndex].first;
    Plot my_plot("State Plot - Chosen EigVal: " + std::to_string(chosenEigVal), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
    my_plot.plotGraph(my_graph);
    my_plot.displayPlot(true);
*/

    // plot initial eigenvalue spectrum and get bin max freq
    int maxFreq = plotHistogramFirst(ev,0);

    // while loop to for each iteration of gradient decent
    int iteration_counter = 1;
    bool exit = false;
    auto eigen_pairs2 = get_eigen_pairs_transpose(B_Matrix);
    while(exit == false){
        std::cout << "\n\nIteration: " << iteration_counter; 

        mean = std::accumulate(ev.begin(), ev.end(), 0.0) / ev.size();
        stdev = calc_stdev(mean, ev);
        std::cout << "\nStarting Variance: " << stdev * stdev;
        //modeIndex = findModeIndex(ev);

        eigen_pairs = eigen_pairs2;        

        // temp graph to for when checking if edge becomes negative during gradient descent iteration
        Graph my_graph2 = my_graph;
        
        double ep = 1 / sqrt(2*(MAX_X*MAX_X) - 2 * MAX_X);
        //double ep = 100;

        // creates vector to hold location of index that gradient should be forced to 0 to prevent negative edge weights
        std::vector<int> neg_index = create_neg_val_index_locations(my_graph);
        
        int counter_edge_index = 0;
        int neg_edge_counter = 0;
        // loops through gradient descent iteration to prevent current iteration from causing negative edge weights
        while(true){
            calc_obj_function(ev, stdev);
            std::cout << "\nTest";
            int counter = 0;
            int counter_edge_index = 0;
            bool neg_detected = false;
            std::vector<double> var_gradient;

            // calculates gradient
            for(int i = 0; i < my_graph.nodes.size(); i++){
                for(int j = 0; j < (my_graph.nodes[i])->neighbors.size(); j++){
                    if((*my_graph.nodes[i]).id < (*my_graph.nodes[i]->neighbors[j]).id){
                        if(neg_index[counter_edge_index] == 0){
                            var_gradient.push_back(0);
                        }
                        else{
                            var_gradient.push_back(compute_var_grad(ev, eigen_pairs[(*my_graph.nodes[i]).id].second, eigen_pairs[(*my_graph.nodes[i]->neighbors[j]).id].second, stdev, false));
                        }
                        counter_edge_index++;     
                    }
                }
            }

            // calculates gradient vector projection
            //std::vector<double> dot_vec = vector_projection();

            std::vector<double> dot_vec = var_gradient;

            // calc adapative gradient value
            double AdaGrad_ep = 1;//calc_adaGrad(dot_vec);
            std::cout << "\nAdagrad: " << AdaGrad_ep;

            // updates edge weights from gradient vector. If edge weights become negative, index is saved and gradient descent is performed again, with that gradient being 0
            neg_edge_counter = 0;
            for(int i = 0; i < my_graph.nodes.size(); i++){
                for(int j = 0; j < (my_graph.nodes[i])->neighbors.size(); j++){
                    if((*my_graph.nodes[i]).id < (*my_graph.nodes[i]->neighbors[j]).id){
                        my_graph2.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) + (ep/(.00000001 + AdaGrad_ep)) * dot_vec[counter];
                        my_graph2.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) + (ep/(.00000001 + AdaGrad_ep)) * dot_vec[counter];
                        if(my_graph2.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) + ep * dot_vec[counter] < 0.00){
                            //std::cout << "\n" << (*my_graph.nodes[i]).id << ", " << (*my_graph.nodes[i]->neighbors[j]).id << " is negative " << my_graph2.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) << " + " << ep * dot_vec[counter] << "\n";
                            neg_detected = true;
                            neg_index[counter] = 0;
                            neg_edge_counter++;
                        }
                        //std::cout << (*my_graph.nodes[i]).id << ", " << (*my_graph.nodes[i]->neighbors[j]).id << ": " << (compute_var_grad(ev, mean, eigen_pairs[(*my_graph.nodes[i]).id].second, eigen_pairs[(*my_graph.nodes[i]->neighbors[j]).id].second)) << "\n";
                        //std::cout << ((*my_graph.nodes[i]).id) << ", " << (*my_graph.nodes[i]->neighbors[j]).id << " - " << dot_vec[counter] << "\n";
                        counter++;
                    }
                }
            }
            if(neg_edge_counter != 0){
                std::cout<< "\nNumber of negative edges: " << neg_edge_counter;
            } 
            if(neg_detected == false){
                int plot = compute_var_grad(ev, eigen_pairs[(*my_graph.nodes[0]).id].second, eigen_pairs[(*my_graph.nodes[0]->neighbors[0]).id].second, stdev, true);
                break;
            }
        }

        // updates graph 
        my_graph = my_graph2;

        
        my_graph.computeMatrices2();
        double trace;
        if(iteration_counter == 1749){
        // compute and print trace and edge sum to ensure graph properties remain the same
        trace = calc_trace(my_graph, MAX_X);

        std::cout << "\nTrace: " << trace;
        
        int scale_factor = (MAX_X*MAX_X*2 - 2 * MAX_X)*2;

        // scale weights
        //my_graph = scale(my_graph, scale_factor, trace);
        int counter = 0;
        for(int i = 0; i < my_graph.nodes.size(); i++){
            for(int j = 0; j < (my_graph.nodes[i])->neighbors.size(); j++){
                if((*my_graph.nodes[i]).id < (*my_graph.nodes[i]->neighbors[j]).id){                              
                    my_graph.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]).id, (*my_graph.nodes[i]->neighbors[j]).id) * (scale_factor/trace);
                    my_graph.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) = my_graph.adjacencyMatrix((*my_graph.nodes[i]->neighbors[j]).id, (*my_graph.nodes[i]).id) * (scale_factor/trace);
                    counter++;
                }
            }
        }
        }
        // recomputes laplacian matrix with new edge weights
        my_graph.computeMatrices2(); 
        Eigen::MatrixXf B_Matrix2 = my_graph.laplacianMatrix;
        eigen_pairs2 = get_eigen_pairs_transpose(B_Matrix2);

        trace = calc_trace(my_graph, MAX_X);
        std::cout << "\nTrace: " << trace << "\n";

        // creates vector of eigenvalues
        std::vector<double> ev2 = create_eigenvalue_vector(eigen_pairs2, false);

        ev = ev2;
        B_Matrix = B_Matrix2;
        iteration_counter++;

        // manual stoppage control
        int N = 1750;
        // display eigenvalue spectrum, final updated graph, and first 4 eigenvectors every N iterations or until can't compute gradient further
        if(iteration_counter % N == 0 || neg_edge_counter == ev.size()){
            plotHistogram(ev, maxFreq, iteration_counter);
            Plot my_plot2("State Plot - Chosen EigVal: " + std::to_string(chosenEigVal), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
            my_plot2.plotGraph(my_graph);
            my_plot2.displayPlot(true);
            auto eigen_pairs_final = get_eigen_pairs(B_Matrix2);
            //plotHeatMap(eigen_pairs_final[0].second, MAX_X, "EV0");
            //plotHeatMap(eigen_pairs_final[1].second, MAX_X, "EV1");
            //plotHeatMap(eigen_pairs_final[2].second, MAX_X, "EV2");
            //plotHeatMap(eigen_pairs_final[3].second, MAX_X, "EV3");
            if(neg_edge_counter == ev.size()){
                break;
            }
            std::cout << "Enter q and press ENTER to exit or press ENTER to continue: ";
            int a = getc(stdin);
            if(a == 113){
                break;
            }
        }
    }


        // automatic
/*
        int N = 1000;
        // display eigenvalue spectrum, final updated graph, and first 4 eigenvectors every N iterations or until can't compute gradient further
        if(iteration_counter % N == 0 || neg_edge_counter == ev.size() || ev[ev.size()-1]>99.8){
            plotHistogram(ev, maxFreq, iteration_counter);
            //Plot my_plot2("State Plot - Chosen EigVal: " + std::to_string(chosenEigVal), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
            //my_plot2.plotGraph(my_graph);
            //my_plot2.displayPlot(true);
            //auto eigen_pairs_final = get_eigen_pairs(B_Matrix2);
            //plotHeatMap(eigen_pairs_final[0].second, MAX_X, "EV0");
            //plotHeatMap(eigen_pairs_final[1].second, MAX_X, "EV1");
            //plotHeatMap(eigen_pairs_final[2].second, MAX_X, "EV2");
            //plotHeatMap(eigen_pairs_final[3].second, MAX_X, "EV3");
            if(neg_edge_counter == ev.size()){
                break;
            }
            if(iteration_counter==25000){
                break;
            }
        }
    }
*/
    // print final variance, mode properties and eigenvalues
    mean = std::accumulate(ev.begin(), ev.end(), 0.0) / ev.size();

    stdev = calc_stdev(mean, ev);

    std::cout << "Final Variance: " << stdev * stdev << "\n";
    modeIndex = findModeIndex(ev);

    print_vec(ev);
    /*
    for(int i = 0; i < MAX_X*MAX_X; i++){
        for(int j = 0; j < MAX_X*MAX_X; j++){
            std::cout << "( " << i << ", " << j << ") :" << my_graph.laplacianMatrix(i,j) << "\n";
        }
    }*/

    /*
    double freq{sqrt(chosenEigVal)};
    Force my_force(amp, freq, my_graph.nodes.size());
    my_force.insertForceElement(1);

    // Generate plot
    for (int i{0}; i < modes.size(); i++)
    {
        Plot eig_plot("Eigen Plot " + std::to_string(i + 1) + " - Eigval: " + std::to_string(eigen_pairs[modes[0]].first), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
        eig_plot.plotGraph(my_graph);
        eig_plot.displayPlot(false);
        eig_plot.plotEigen(my_graph, eigen_pairs[modes[0]].second);
        eig_plot.displayPlot(true);
    }
    */
    /*
    // Simulate dynamics
    int simulationTime{400};
    int simulationSteps{simulationTime * 100};
    Dynamics my_sim(simulationTime, simulationSteps, damping, stiffness);
    if (decentralizedAlg)
        my_sim.runDecentralizedDynamics(my_graph.nodes, my_force, my_plot);
    else
        my_sim.runCentralizedDynamics(my_graph, my_force, my_plot);
    my_plot.displayPlot(true);*/

    return 0;
}