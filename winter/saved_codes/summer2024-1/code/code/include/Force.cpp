#define _USE_MATH_DEFINES

#include "Force.hpp"
#include <cmath>
#include <iostream>

Force::Force(double amp, int size) : amplitude{amp}, size{size} {}

void Force::insertForceElement(int idx)
{
    nonZeroElements.push_back(idx);
}
/*
Eigen::VectorXf Force::sinusoidalForce(double t)
{
    Eigen::VectorXf force(size);
    for (int i{0}; i < size; i++)
    {
        force(i) = 0;
    }
    for (int i{0}; i < nonZeroElements.size(); i++)
    {
        int idx = nonZeroElements[i];
        force(idx) = amplitude * sin(frequency * t);
    }
    return force;
}*/

/*
Eigen::VectorXf Force::sinCauchyForce(double t)
{
    srand(seedNum);
    Eigen::VectorXf force(size);
    for (int i{0}; i < size; i++)
    {
        force(i) = 0;
    }
    std::vector<double> freq_vec;
    for(int i = 0; i < 10; i++){
        freq_vec.push_back(((double)rand() / (double)RAND_MAX));
    }
    double func = 0;
    for(int i = 0; i < 10; i++){
        //func = func + sin(2*M_PI*frequency*freq_vec[i]*t);
        func = func + sin(inverse_of_normal_cdf(freq_vec[i], frequency, alpha * frequency)*t);
        //std::cout << inverse_of_normal_cdf(freq_vec[i],4.1,0.005*2) << " " <<  freq_vec[i] << "\n";       //func = func + sin(frequency*freq_vec[i]*t);
    }
    for (int i{0}; i < nonZeroElements.size(); i++)
    {
        int idx = nonZeroElements[i];
        force(idx) = amplitude*func;
    }
    return force;
}*/

Eigen::VectorXf Force::sinCauchyForceNew(double t, double freq_used)
{
    //srand(seedNum);
    Eigen::VectorXf force(size);
    for (int i{0}; i < size; i++)
    {
        force(i) = 0;
    }

    //double freq = (((double)rand() / (double)RAND_MAX));

 //std::cout << "Sampled Freq: " << freq << "\n";

    double func = sin(freq_used*t);

    for (int i{0}; i < nonZeroElements.size(); i++)
    {
        int idx = nonZeroElements[i];
        force(idx) = amplitude*func;
    }
    return force;
}