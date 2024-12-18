#ifndef _FORCE_H_
#define _FORCE_H_

#include <vector>
#include <Eigen/Dense>

class Force
{
public:
    double amplitude;
    int size;
    std::vector<int> nonZeroElements;

    Force(double amp, int size);

    void insertForceElement(int idx);

    Eigen::VectorXf sinusoidalForce(double t);
    Eigen::VectorXf sinCauchyForce(double t);
    Eigen::VectorXf sinCauchyForceNew(double t, double freq_used);

};

#endif