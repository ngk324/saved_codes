#ifndef _GRAPH_H_
#define _GRAPH_H_

#include "Node.hpp"
#include <Eigen/Dense>

#include "gdMain.hpp"

class Graph
{
private:
public:
    std::vector<std::shared_ptr<Node>> nodes;
    Eigen::MatrixXf adjacencyMatrix, degreeMatrix, laplacianMatrix;
    Eigen::MatrixXf connectivityMatrix;
    std::vector<double> eigenValues;
    //Eigen::VectorXf eigenValues;
    //Eigen::MatrixXf eigenVectors;

    static constexpr double eps{0.1};

    void computeEigenValues(std::string filename);

    void constructSimpleGraph();

    void setNeighbors();

    void computeMatrices(std::string fileName);
    //void eigenDecompose();

    void simulateDynamics(const int tMax);
};

#endif
