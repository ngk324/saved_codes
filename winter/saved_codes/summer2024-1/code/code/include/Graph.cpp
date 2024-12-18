#include "Graph.hpp"
#include <iostream>
#include <memory>
#include <numeric>
#include "gdMain.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <string>

void Graph::constructSimpleGraph()
{
    // Randomly assign some state value and generate nodes
    srand(time(NULL));
    //srand(484);
    int idx{0};
    for (int i{0}; i < numNodes; i++)
    {
        double rand_state{((double)rand() / (double)RAND_MAX)};
        std::shared_ptr<Node> ptr{new Node(idx, rand_state*.25)};
        nodes.push_back(ptr);
        idx++;
    }
}

void Graph::computeMatrices(std::string filename)
{
    std::vector<double> buffer;
    std::string line;
    std::ifstream myFile(filename);
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
    laplacianMatrix = L;
    degreeMatrix = D;
    connectivityMatrix = C;
    adjacencyMatrix = A;
    //eigenDecompose();
}

void Graph::computeEigenValues(std::string filename)
{
    std::vector<double> buffer;
    std::string line;
    std::ifstream myFile(filename);
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
       eigenValues.push_back(buffer[i]);
    }
    //eigenDecompose();
}

/*
void Graph::eigenDecompose()
{
    Eigen::MatrixXf matrixToSolve = laplacianMatrix + eps * Eigen::MatrixXf::Identity(laplacianMatrix.rows(), laplacianMatrix.cols());
    Eigen::EigenSolver<Eigen::MatrixXf> solver(matrixToSolve);
    eigenValues = solver.eigenvalues().real();
    eigenVectors = solver.eigenvectors().real();

    // Combine eigenvalues and eigenvectors into a std::vector of pairs
    std::vector<std::pair<double, Eigen::VectorXf>> eigenPairs;
    for (int i = 0; i < eigenValues.size(); ++i) {
        eigenPairs.push_back(std::make_pair(eigenValues[i], eigenVectors.col(i)));
    }

    // Sort the vector of pairs based on eigenvalues in ascending order
    std::sort(eigenPairs.begin(), eigenPairs.end(), [](const std::pair<double, Eigen::VectorXf>& a,
                const std::pair<double, Eigen::VectorXf>& b) {
            return a.first < b.first;
            });

    // Extract the sorted eigenvalues and eigenvectors
    for (int i = 0; i < eigenValues.size(); ++i) {
        eigenValues[i] = eigenPairs[i].first;
        eigenVectors.col(i) = eigenPairs[i].second;
    }
}*/


