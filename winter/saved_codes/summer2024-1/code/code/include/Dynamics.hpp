#ifndef _DYNAMICS_H_
#define _DYNAMICS_H_

#include "Graph.hpp"
#include "Plot.hpp"
#include "Force.hpp"

#include <memory>

#include "EnergyPlot.hpp"
#include "gnuplot-iostream.h"

class Dynamics
{
public:
    int simTime;
    int simSteps;
    double dampingCoeff;
    double stiffnessCoeff;
    double epsilon;
    int simNum;

    Dynamics(int sim_time, int sim_steps, double damping, double stiffness, double epsilon, int simNum);

    Eigen::VectorXf getStateVector(Graph &g) const;
    void setNodeStates(Graph &g, Eigen::VectorXf &states) const;
    void write_SS(std::vector<std::complex<double>> SS);

    friend void ode_system(Graph &g, Force &force, double freq_used, const Eigen::VectorXf &x, Eigen::VectorXf &dxdt, const double t);

    void writeNodeAvgFile(std::vector<double> nodeValsMax, double avg);
    void writeTwoNormAvgFile(double avg);
    void writeNodeValuesFile(std::vector<std::vector<double>> XValueHistory, int nodeSize, int simSteps);
    double calculateTwoNormVals(std::vector<std::complex<double>> XValueHistory, int startTime, int windowSize);
    double inverse_of_normal_cdf(const double p, const double mu, const double sigma);
    void write_file_results(std::string print);
    void write_freq_file(std::vector<double> freq);
    void write_two_norm_file_results(double twoNormAvg);
    std::vector<double> get_force_vec(std::string filename);
    std::vector<double> get_freqs(std::string filename);

    void runCentralizedDynamics(Graph &g, Force &force, Plot &plot);
    void runDecentralizedDynamics(std::vector<std::shared_ptr<Node>> &nodes, Force &force, Plot &plot) const;
    std::vector<std::complex<double>> calculateNodeVals(std::vector<std::vector<std::complex<double>>> XHistory, int startTime, int windowSize);
    bool determineSteadyState(std::vector<double> energyValueHistory, int iterationRange, double percentDifference);
    Gnuplot energyPlotStream;
    std::vector<double> energyValueHistory;
    std::vector<double> XValueHistory1;

};

#endif
