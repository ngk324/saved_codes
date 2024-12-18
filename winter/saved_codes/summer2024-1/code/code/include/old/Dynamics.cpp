#include "Dynamics.hpp"
#include "EnergyPlot.hpp"
#include "XPlot.hpp"
#include "TwoNormPlot.hpp"

#include <unistd.h>
#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>
#include <string>
#include <string.h>
#include <iostream>
#include <random>

#include <complex>
#include <cmath>

#include "gdMain.hpp"

#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

Dynamics::Dynamics(int sim_time, int sim_steps, double damping, double stiffness, double epsilon)
    : simTime{sim_time}, simSteps{sim_steps}, dampingCoeff{damping}, stiffnessCoeff{stiffness}, epsilon{epsilon} {}

Eigen::VectorXf Dynamics::getStateVector(Graph &g) const
{
    int nNodes = g.nodes.size();
    Eigen::VectorXf vec(nNodes);
    for (int i{0}; i < nNodes; i++)
    {
        vec(i) = g.nodes[i]->z;
    }
    return vec;
}

void Dynamics::setNodeStates(Graph &g, Eigen::VectorXf &states) const
{
    for (int i{0}; i < states.size(); i++)
    {
        g.nodes[i]->z_old = g.nodes[i]->z;
        g.nodes[i]->z = states(i);
    }
}

void Dynamics::writeNodeAvgFile(std::vector<double> nodeValsMax, double avg){
    std::string fileName = "NodeAvgResults";
    fileName.append(std::to_string(simNum));
    fileName = fileName + "-";
    fileName.append(std::to_string(!beforeGrad));
    fileName = fileName + ".txt";
    std::string line;
    std::ofstream myFile(fileName);
    for(int i{0}; i <= nodeValsMax.size(); i++){
        if(i < nodeValsMax.size()){
            myFile << "Node " << i << ": " << nodeValsMax[i] << "\n";
        }
        else{
            myFile << "Node Avg: " << ": " << avg;
        }
    }
    myFile.close();
}

void Dynamics::writeTwoNormAvgFile(double avg){
    std::string fileName = "TwoNormResults";
    fileName.append(std::to_string(simNum));
    fileName = fileName + "-";
    fileName.append(std::to_string(!beforeGrad));
    fileName = fileName + ".txt";
    std::string line;
    std::ofstream myFile(fileName);
    myFile << "Two Norm Avg: " << avg;
    myFile.close();
}

void Dynamics::write_two_norm_file_results(double twoNormAvg){
    std::string fileName;
    if(beforeGrad){
        fileName = "ONLY-BEFORE-TwoNormResults.txt";
    }
    else{
        fileName = "ONLY-AFTER-TwoNormResults.txt";
    }
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    log << std::to_string(twoNormAvg) << std::endl;
    log.close();
}

void Dynamics::writeNodeValuesFile(std::vector<std::vector<double>> XValueHistory, int nodeSize, int simSteps){
    std::string fileName = "NodeValueResults";
    fileName.append(std::to_string(simNum));
    fileName = fileName + "-";
    fileName.append(std::to_string(!beforeGrad));
    fileName = fileName + ".txt";
    std::string line;
    std::ofstream myFile(fileName);

    for(int j{0}; j < nodeSize; j++){
        myFile << "Node " << j << ":\n";
        for (int i{0}; i < simSteps; i++)
        {
            if(i % 99 == 0){
                myFile << XValueHistory[j][i] << ", ";
            }
        }
        myFile << "\n\n\n";
    }
    myFile.close();
}

double Dynamics::inverse_of_normal_cdf(const double p, const double mu, const double sigma){
    double inverse = mu + sigma * tan(M_PI*(p - .5));
    return inverse;
}

void Dynamics::write_file_results(std::string print){
    std::string fileName = "AAA-TOTAL-TwoNormResults.txt";
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    log << "\n" << print;
    log.close();
}

typedef std::vector<std::complex<double>> state_type;


struct push_back_state_time
{
    std::vector<state_type> &m_states;
    std::vector<double> &m_times;
    std::vector<std::complex<double>> &twoNorm;
    std::complex<double> type = 0.0;


    push_back_state_time(std::vector<state_type> &states, std::vector<double> &times, std::vector<std::complex<double>> &twoNorm) : m_states(states), m_times(times), twoNorm(twoNorm) {}

    void operator()(const state_type &x, double t)
    {
        m_states.push_back(x);
        m_times.push_back(t);
        twoNorm.push_back(std::inner_product(x.begin(), x.begin()+101, x.begin(), type));
        int time = t * 100;
        if(time > 100000){
            if(abs(std::abs(twoNorm[time-1001]) - std::abs(twoNorm[time-1])) < 0.0001 && abs(std::abs(twoNorm[time-501]) - std::abs(twoNorm[time-1])) < 0.0001){
            //if(abs(std::abs(twoNorm[time-1001]) - std::abs(twoNorm[time-1])) < 0.000001 && abs(std::abs(twoNorm[time-501]) - std::abs(twoNorm[time-1])) < 0.000001){
                throw std::runtime_error( "Too much steps" );
            }
        }
    }
};

class secondODE
{
    Eigen::MatrixXf D;
    Eigen::MatrixXf L;
    state_type force;
    double freq_used;

public:
    //secondODE(std::vector<double> D, std::vector<double> L, double freq_used) : D(D), L(L), freq_used(freq_used) {}

    secondODE(Eigen::MatrixXf &D, Eigen::MatrixXf &L, state_type force, double freq_used): D(D), L(L), force(force), freq_used(freq_used) {}

    void operator()(const state_type &x, state_type &dxdt, const double t)
    {
        const std::complex<double> ci(0.0,1.0); 
        size_t N = numNodes;
        state_type ss(N);
        for( size_t i=0 ; i<N ; i++ )
        {
            std::complex<double> sum = 0;
            for( size_t j=0 ; j<N ; j++ )
            {
                sum += (-D(i,j)*x[j+N].real() - L(i,j)*x[j].real());
            }
            dxdt[i] = x[i+N];
            dxdt[i+N] = sum;
            if(force[i].real() != 0){
                dxdt[i+N] += force[i]*sin(freq_used*t);
               //dxdt[i+N] += force[i]*std::exp(ci*freq_used*t);
            }
        }
    
    }

};

//typedef runge_kutta_cash_karp54<state_type> error_stepper_type;
//typedef controlled_runge_kutta<error_stepper_type> controlled_stepper_type;


void Dynamics::runCentralizedDynamics(Graph &g, Force &force, Plot &plot)
{
    plot.displayMethod("Centralized");
    int nNodes = g.nodes.size();
    Eigen::MatrixXf A_Matrix = dampingCoeff * (Eigen::MatrixXf::Identity(nNodes, nNodes));
    Eigen::MatrixXf B_Matrix = (g.laplacianMatrix + epsilon * Eigen::MatrixXf::Identity(nNodes, nNodes));
    //Eigen::MatrixXf B_Matrix = (g.laplacianMatrix);
    Eigen::VectorXf x = getStateVector(g);
    Eigen::VectorXf x_dot = Eigen::VectorXf::Zero(nNodes);
    Eigen::VectorXf x_ddot(nNodes);
    double timeStep = double(simTime) / simSteps;    

    //std::vector<double> XValueHistory[g.nodes.size()][simSteps];
    std::vector<std::vector<double>> XValueHistory(g.nodes.size(), std::vector<double> (simSteps, 0.0));
    std::vector<std::complex<double>> twoNorm;

    std::cout << "Prob used for sampling: " << randProb << std::endl;
    std::string probUsed = "Prob used for sampling: " + std::to_string(randProb);
    write_file_results(probUsed);

    double freq_used = inverse_of_normal_cdf(randProb, frequencyFromUniform, h);
    std::cout << "Freq from sampling: " << freq_used << std::endl;
    std::string freqFromSample = "Freq from sampling: " + std::to_string(freq_used);
    write_file_results(freqFromSample);

    size_t N = numNodes;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);

    std::vector<double> force_vec;

    // sample from unit sphere and fill force vec
    for (int i=0; i<numNodes; ++i) {
        double number = distribution(generator);
        force_vec.push_back(number);
    }
    double norm = sqrt(std::inner_product(force_vec.begin(), force_vec.begin()+101, force_vec.begin(), 0.0));
    for (int i=0; i<force_vec.size(); ++i) {
        force_vec[i] = force_vec[i] / norm;
    }

    norm = sqrt(std::inner_product(force_vec.begin(), force_vec.begin()+101, force_vec.begin(), 0.0));

    std::vector<std::complex<double>> freqUsed(N);
    for (int i{0}; i < N; i++)
    {
        freqUsed[i] = force_vec[i];//*sqrt(freq_used);
    }
    std::cout << freq_used << std::endl;
    
   
    state_type x1(2*N);
    for(int i = 0; i < N; i++){
        x1[i] = x[i];
        x1[i+N] = 0.0;
    }

    std::vector<std::vector<std::complex<double>>> x_vec;
    std::vector<double> times;
    /*
    double abs_err = 1.0e-10, rel_err = 1.0e-6, a_x = 1.0, a_dxdt = 1.0;
   controlled_stepper_type controlled_stepper(
        default_error_checker<double, range_algebra, default_operations>(abs_err, rel_err, a_x, a_dxdt));*/
    
    secondODE result(A_Matrix,B_Matrix,freqUsed,freq_used);
    runge_kutta4<state_type> stepper;
    double t_start = 0.0 , t_end = 10000.0 , dt = 0.01;
    //size_t steps = integrate_adaptive(controlled_stepper, result , x1 , t_start , t_end , dt , push_back_state_time(x_vec, times));
    
    size_t steps = 0;

    try {
        steps = integrate_const(stepper,result , x1 , t_start , t_end , dt , push_back_state_time(x_vec, times, twoNorm));
    }
    catch(...){
        //std::cout << "breakout" << std::endl;
        steps = std::size(twoNorm) - 2;
    }
    std::complex<double> type = 0.0;
    //for (size_t i = 0; i <= steps; i++)
    //{
        //std::cout << times[i] << '\t' << x_vec[i][2] << '\t' << x_vec[i][199] << '\n';
        //twoNorm.push_back(std::inner_product(x_vec[i].begin(), x_vec[i].begin()+101, x_vec[i].begin(), type));
    //}



    std::cout << std::inner_product(x_vec[steps].begin(), x_vec[steps].begin()+101, x_vec[steps].begin(),type) << std::endl;
/*
    for (int i{0}; i < simSteps + 1; i++)
    {
        //x_ddot = force.sinCauchyForceNew(i * timeStep) - A_Matrix * x_dot - B_Matrix * x;
        x_ddot = force.sinCauchyForceNew(i * timeStep, freq_used) - A_Matrix * x_dot - B_Matrix * x;
        x_dot += (x_ddot * timeStep);
        x += (x_dot * timeStep);

        double energyVal = (.5 * x_dot.transpose()*Eigen::MatrixXf::Identity(nNodes, nNodes)*x_dot);
        energyVal += (.5 * x.transpose()*g.laplacianMatrix*x); 

        energyValueHistory.push_back(energyVal);

        for(int j{0}; j < g.nodes.size(); j++){
            XValueHistory[j][i] = (double)x[j];
        }

        twoNorm.push_back((double)std::inner_product(x.begin(), x.end(), x.begin(), 0.0));

        //XValueHistory1.push_back(x[1]);

        setNodeStates(g, x);

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
        for (int j{0}; j < nNodes; j++)
        {
            plot.plotNodeCircle(*g.nodes[j], maxZ, minZ,j);
            plot.displayState(*g.nodes[j],j);
        }
        // std::cout << x << std::endl
        //           << std::endl;
        plot.displayTime(std::to_string(i * timeStep) + " s");
        plot.displayPlot();
        usleep(1E+3 * timeStep);
        /*if(i > 25000){
            std::cout << "\n" << 100 * ((energyValueHistory[energyValueHistory.size() - 25000] - energyValueHistory.back()) / energyValueHistory.back());
            //energyPlot::generateEnergyPlot(energyPlotStream, energyValueHistory);
        }*/
    //}
    
    /*int numOfWindows = 5;
    int widthOfWindow = 200000;
    int startTime = 9000000;*/
    /*int numOfWindows = 1;
    int widthOfWindow = 1000;
    int startTime = 1000;*/
    /*double numOfWindows = 5.0;
    int widthOfWindow = 20000;
    int startTime = 650000;*/
    double numOfWindows = 5.0;
    int widthOfWindow = 20000;
    int startTime = steps-100000;

    std::vector<double> nodeValsMax(numNodes, 0);
    double twoNormAvg = 0;
    for(int i{0}; i < numOfWindows; i++){
        std::vector<std::complex<double>> nodeMaxVector = calculateNodeVals(x_vec, startTime + i * widthOfWindow, widthOfWindow);
        twoNormAvg += (calculateTwoNormVals(twoNorm, startTime + i * widthOfWindow, widthOfWindow) / (double) numOfWindows);
        for(int j{0}; j < nodeValsMax.size(); j++){
            nodeValsMax[j] += (nodeMaxVector[j].real()/ (numOfWindows));
        }
    }
    double avg = 0;
    for(int i{0}; i <= numNodes; i++){
        if(i < numNodes){
            avg += nodeValsMax[i];
        }
        else{
            avg = avg / (double)numNodes;
        }
    }
   // writeNodeAvgFile(nodeValsMax, avg);
    //writeTwoNormAvgFile(twoNormAvg);
    //writeNodeValuesFile(XValueHistory, g.nodes.size(), simSteps+1);
    write_two_norm_file_results(twoNormAvg);
    std::string twoNormVal = "Two Norm Avg: " + std::to_string(twoNormAvg);
    write_file_results(twoNormVal);

    //energyPlot::generateEnergyPlot(energyPlotStream, energyValueHistory);
    //XPlot::generateXPlot(energyPlotStream, XValueHistory1);
    //twoNormPlot::generateTwoNormPlot(energyPlotStream, twoNorm);
}

std::vector<std::complex<double>> Dynamics::calculateNodeVals(std::vector<std::vector<std::complex<double>>> XValueHistory, int startTime, int windowSize){ // end time = numOfWindows*windowSize + startTime
    std::vector<std::complex<double>> nodeMax(numNodes, 0);
    for(int i = startTime; i < startTime + windowSize; i++){
        for(int j{0}; j < numNodes; j++){
            if(nodeMax[j].real() < abs((XValueHistory[j][i]).real())){
                nodeMax[j] = abs((XValueHistory[j][i]).real());
            }
        }
    }
    return nodeMax;
}

double Dynamics::calculateTwoNormVals(std::vector<std::complex<double>> XValueHistory, int startTime, int windowSize){ // end time = numOfWindows*windowSize + startTime
    double nodeMax = 0;
    for(int i = startTime; i < startTime + windowSize; i++){
        if(nodeMax < abs(XValueHistory[i].real())){
            nodeMax = abs(XValueHistory[i].real());
        }
    }
    return nodeMax;
}

bool Dynamics::determineSteadyState(std::vector<double> energyValueHistory, int iterationRange, double percentDifference){
    bool withinPercent = false;
    double changeFromRange = ((energyValueHistory[energyValueHistory.size() - iterationRange - 1] - energyValueHistory.back()) / energyValueHistory.back());
    double changeFromPrevious = ((energyValueHistory[energyValueHistory.size() - 2] - energyValueHistory.back()) / energyValueHistory.back());
    if((changeFromRange - changeFromPrevious) / changeFromPrevious < 0.1){
        withinPercent = true;
    }
    return withinPercent;
}
/*
void Dynamics::runDecentralizedDynamics(std::vector<std::shared_ptr<Node>> &nodes, Force &force, Plot &plot) const
{
    plot.displayMethod("Decentralized");
    double timeStep = double(simTime) / simSteps;
    for (int i{0}; i < simSteps + 1; i++)
    {
        Eigen::VectorXf force_vec = force.sinusoidalForce(i * timeStep);
        for (int j{0}; j < nodes.size(); j++)
        {
            double neighbor_z_sum{0}, neighbor_zdot_sum{0};
            for (int k{0}; k < nodes[j]->neighbors.size(); k++)
            {
                neighbor_z_sum += nodes[j]->neighbors[k]->z_old;
                neighbor_zdot_sum += nodes[j]->neighbors[k]->z_dot_old;
            }
            double z_ddot = force_vec(j) - dampingCoeff * (nodes[j]->neighbors.size() * nodes[j]->z_dot - neighbor_zdot_sum + epsilon * nodes[j]->z_dot) - stiffnessCoeff * (nodes[j]->neighbors.size() * nodes[j]->z - neighbor_z_sum + epsilon * nodes[j]->z);
            nodes[j]->z += (nodes[j]->z_dot * timeStep);
            nodes[j]->z_dot += (z_ddot * timeStep);
        }
        for (int j{0}; j < nodes.size(); j++)
        {
            nodes[j]->z_old = nodes[j]->z;
            nodes[j]->z_dot_old = nodes[j]->z_dot;
            //plot.plotNode(*nodes[j]);
            plot.displayState(*nodes[j]);
        }

        plot.displayTime(std::to_string(i * timeStep) + " s");
        plot.displayPlot();
        usleep(1E+2 * timeStep);
    }
}*/
