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

Dynamics::Dynamics(int sim_time, int sim_steps, double damping, double stiffness, double epsilon, int simNum)
    : simTime{sim_time}, simSteps{sim_steps}, dampingCoeff{damping}, stiffnessCoeff{stiffness}, epsilon{epsilon}, simNum{simNum} {}

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
        fileName = "test-b-social/AAA_ONLY-BEFORE-TwoNormResults.txt";
    }
    else{
        fileName = "test-a-social/AAA_ONLY-AFTER-TwoNormResults.txt";
    }
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    log << std::to_string(twoNormAvg) << std::endl;
    log.close();
}

void Dynamics::write_SS(std::vector<std::complex<double>> SS){
    std::string fileName;
    if(beforeGrad){
        fileName = "test-b-social/Amp-Before-";
        fileName.append(std::to_string(simNum));
        fileName = fileName + ".txt";
    }
    else{
        fileName = "test-a-social/Amp-After-";
        fileName.append(std::to_string(simNum));
        fileName = fileName + ".txt";
    }
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    for(int i = 0; i < SS.size(); i++){
        //log << std::to_string(SS[i].real()) << std::endl;
        double to_print = sqrt(norm(SS[i]));
        log << std::to_string(to_print) << std::endl;
        //log << std::to_string(SS[i]) << std::endl;
    }
    std::cout << "Complex in Print File " << SS.back() << std::endl;
    std::cout << "SQRTNORM in Print File " << sqrt(norm(SS.back())) << std::endl;
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

void Dynamics::write_freq_file(std::vector<double> freq){
    std::string fileName = "freqs";
    fileName.append(std::to_string(simNum));
    fileName = fileName + "-";
    fileName.append(std::to_string(!beforeGrad));
    fileName = fileName + ".txt";
    std::string line;
    std::ofstream myFile(fileName);

    for(int i{0}; i < freq.size(); i++){
        myFile << freq[i];
        myFile << "\n";
    }
    myFile << "\n\n\n";
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
    std::complex<double> type; 
    double freq_used;
    

    push_back_state_time(std::vector<state_type> &states, std::vector<double> &times, std::vector<std::complex<double>> &twoNorm, double freq_used) : m_states(states), m_times(times), twoNorm(twoNorm), freq_used(freq_used) {}

    void operator()(const state_type &x, double t)
    {
        m_states.push_back(x);
        m_times.push_back(t);
        const std::complex<double> ci(0.0,1.0); 
        std::vector<std::complex<double>> complConj;

        for(int i = 0; i < numNodes; i++){
            complConj.push_back(std::conj(x[i]));
        }
        twoNorm.push_back(std::inner_product(x.begin(), x.begin()+1+numNodes, complConj.begin(), std::complex<double>(0.,0.)));
        //std::cout << twoNorm.back() << std::endl;
        int time = t * 100;
        if(time > 100000){
            int window_size = ceil((2.0 * 3.14 * 100) / abs(freq_used));
            double window1 = sqrt(norm(std::accumulate(twoNorm.begin()+time-4*window_size,twoNorm.begin()+time-3*window_size,std::complex<double>(0.,0.)))) / window_size;
            double window2 = sqrt(norm(std::accumulate(twoNorm.begin()+time-3*window_size,twoNorm.begin()+time-2*window_size,std::complex<double>(0.,0.)))) / window_size;
            double window3 = sqrt(norm(std::accumulate(twoNorm.begin()+time-2*window_size,twoNorm.begin()+time-window_size,std::complex<double>(0.,0.)))) / window_size;
            double window4 = sqrt(norm(std::accumulate(twoNorm.begin()+time-window_size,twoNorm.begin()+time,std::complex<double>(0.,0.)))) / window_size;
            double perDif = 0.00001;
            if(abs(window1 - window4)/window1 < perDif && abs(window1 - window4)/window4 < perDif && (window2 - window4)/window2 < perDif && abs(window2 - window4)/window2 < perDif && (window3 - window4)/window3 < perDif && abs(window3 - window4)/window4 < perDif){
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
    secondODE(Eigen::MatrixXf &D, Eigen::MatrixXf &L, state_type force, double freq_used): D(D), L(L), force(force), freq_used(freq_used) {}

    void operator()(const state_type &x, state_type &dxdt, const std::complex<double> t)

    {
        const std::complex<double> ci(0.0,1.0); 
        size_t N = numNodes;
        state_type ss(N);
        for( size_t i=0 ; i<N ; i++ )
        {
            std::complex<double> sum = 0;
            for( size_t j=0 ; j<N ; j++ )
            {   
                std::complex<double> Damp(-D(i,j),0); 
                std::complex<double> Lap(-L(i,j),0); 
                sum += (Damp * x[j+N]) + Lap*(x[j]);
            }
            dxdt[i] = x[i+N];
            dxdt[i+N] = sum;
            std::complex<double> freqq(freq_used,0);
            dxdt[i+N] += force[i]*(cos(freqq*t) + ci*sin(freqq*t));
            //dxdt[i+N] += force[i]*exp(ci*freqq*t);
        }
    
    }

};

std::vector<double> Dynamics::get_force_vec(std::string filename){
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
    std::vector<double> forces;
    for(int i = 0; i < buffer.size(); i++){
        forces.push_back(buffer[buffer.size() - 1 - i]);
        //forces.push_back(buffer[i]);
    }
    return buffer;
}

std::vector<double> Dynamics::get_freqs(std::string filename){
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
    return buffer;
}


void Dynamics::runCentralizedDynamics(Graph &g, Force &force, Plot &plot)
{
    plot.displayMethod("Centralized");
    int nNodes = g.nodes.size();
    double ep = 1;
    double gamma = 0.001;
    std::vector<std::vector<double>> XValueHistory(g.nodes.size(), std::vector<double> (simSteps, 0.0));
    std::vector<std::complex<double>> twoNorm;
/*
    std::cout << "Prob used for sampling: " << randProb << std::endl;
    std::string probUsed = "Prob used for sampling: " + std::to_string(randProb);
    write_file_results(probUsed);

    double freq_used = inverse_of_normal_cdf(randProb, frequencyFromUniform, h);
    std::cout << "Freq from sampling: " << freq_used << std::endl;
    std::string freqFromSample = "Freq from sampling: " + std::to_string(freq_used);
    write_file_results(freqFromSample);
    */
    size_t N = numNodes;

/*
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
*/
    std::string filename_force = "";
    std::string filename_freqs = "";
    std::vector<double> force_vec;
    std::vector<double> freqs;
    std::vector<double> buffer;
    std::string line;
    Eigen::MatrixXf OMG(numNodes, numNodes);
    if(beforeGrad){
        filename_force = "42/forcing_vec_initial" + std::to_string(simNum) + ".txt";
        filename_freqs = "42/forcing_freq_initial.txt";
        std::ifstream myFile("OMG_Initial_62.txt");
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
                OMG(i,j) = buffer[counter];
                counter++;
            }
        }
    }
    else{

        filename_force = "42/forcing_vec_final" + std::to_string(simNum) + ".txt";
        filename_freqs = "42/forcing_freq_final.txt";
        std::ifstream myFile("OMG_Final_62.txt");

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
                OMG(i,j) = buffer[counter];
                counter++;
            }
        }
    }
    force_vec = get_force_vec(filename_force);

    freqs = get_freqs(filename_freqs);

    double freq_used = freqs[simNum-1];
    printf("%f\n",freq_used);


    Eigen::MatrixXf A_Matrix = 2 * gamma * (g.laplacianMatrix + ep * Eigen::MatrixXf::Identity(nNodes, nNodes));
    Eigen::MatrixXf B_Matrix = (g.laplacianMatrix + ep * Eigen::MatrixXf::Identity(nNodes, nNodes));
    Eigen::VectorXf x = getStateVector(g);
    Eigen::VectorXf x_dot = Eigen::VectorXf::Zero(nNodes);
    Eigen::VectorXf x_ddot(nNodes);
    double timeStep = double(simTime) / simSteps;   



/*
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
*/

    std::vector<std::complex<double>> freqUsed(N);
    for (int i{0}; i < N; i++)
    {
        freqUsed[i] = force_vec[i];//*sqrt(freq_used);
    }
    //std::cout << freq_used << std::endl;
    //write_freq_file(force_vec);
    
   
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
    double t_start = 0.0 , t_end = simTime , dt = 0.01;
    //size_t steps = integrate_adaptive(controlled_stepper, result , x1 , t_start , t_end , dt , push_back_state_time(x_vec, times));
    
    size_t steps = 0;

    try {
        steps = integrate_const(stepper,result , x1 , t_start , t_end , dt , push_back_state_time(x_vec, times, twoNorm, freq_used));
    }
    catch(...){
        std::cout << "breakout" << std::endl;
        steps = std::size(twoNorm) - 2;
    }
    std::complex<double> type(0.0,0.0);

    std::cout << steps << std::endl;



    //std::cout << std::inner_product(x_vec[steps].begin(), x_vec[steps].begin()+101, x_vec[steps].begin(),type) << std::endl;
    
    /*
    double numOfWindows = 5.0;
    int widthOfWindow = 2000;
    int startTime = steps-10000;
    */
    /*int numOfWindows = 5.0;
    int widthOfWindow = 5000;
    int startTime = steps-numOfWindows*widthOfWindow;
    std::vector<double> nodeValsMax(numNodes, 0);
    double twoNormAvg = 0;
    for(int i{0}; i < numOfWindows; i++){
        std::vector<std::complex<double>> nodeMaxVector = calculateNodeVals(x_vec, startTime + i * widthOfWindow, widthOfWindow);
        twoNormAvg += (calculateTwoNormVals(twoNorm, startTime + i * widthOfWindow, widthOfWindow) / (double) numOfWindows);
        for(int j{0}; j < nodeValsMax.size(); j++){
            //nodeValsMax[j] += (nodeMaxVector[j].real()/ (numOfWindows));
            nodeValsMax[j] += (sqrt(norm(nodeMaxVector[j]))/ (numOfWindows));

        }
    }*/
    write_SS(twoNorm);

    int numOfWindows = 4.0;
    int window_size = ceil((2.0 * 3.14 * 100) / abs(freq_used));
    int startTime = steps-numOfWindows*window_size;
    
    double window1 = sqrt(norm(std::accumulate(twoNorm.begin()+startTime-4*window_size,twoNorm.begin()+startTime-3*window_size,std::complex<double>(0.,0.)))) / window_size;
    double window2 = sqrt(norm(std::accumulate(twoNorm.begin()+startTime-3*window_size,twoNorm.begin()+startTime-2*window_size,std::complex<double>(0.,0.)))) / window_size;
    double window3 = sqrt(norm(std::accumulate(twoNorm.begin()+startTime-2*window_size,twoNorm.begin()+startTime-window_size,std::complex<double>(0.,0.)))) / window_size;
    double window4 = sqrt(norm(std::accumulate(twoNorm.begin()+startTime-window_size,twoNorm.begin()+startTime,std::complex<double>(0.,0.)))) / window_size;
    double twoNormAvg = (window1 + window2 + window3+window4)/4;
    std::cout << "After windowed average Complex " << twoNorm.back() << std::endl;
    std::cout << "After windowed average SQRTNORM " << sqrt(norm(twoNorm.back())) << std::endl;
    /*
    double avg = 0;
    for(int i{0}; i <= numNodes; i++){
        if(i < numNodes){
            avg += nodeValsMax[i];
        }
        else{
            avg = avg / (double)numNodes;
        }
    }*/
   // writeNodeAvgFile(nodeValsMax, avg);
    //writeTwoNormAvgFile(twoNormAvg);
    //writeNodeValuesFile(XValueHistory, g.nodes.size(), simSteps+1);
    write_two_norm_file_results(twoNormAvg);
    //std::string twoNormVal = "Two Norm Avg: " + std::to_string(twoNormAvg);
    //write_file_results(twoNormVal);

    //energyPlot::generateEnergyPlot(energyPlotStream, energyValueHistory);
    //XPlot::generateXPlot(energyPlotStream, XValueHistory1);
    //twoNormPlot::generateTwoNormPlot(energyPlotStream, twoNorm);
    twoNorm.clear();
}

std::vector<std::complex<double>> Dynamics::calculateNodeVals(std::vector<std::vector<std::complex<double>>> XValueHistory, int startTime, int windowSize){ // end time = numOfWindows*windowSize + startTime
    std::vector<std::complex<double>> nodeMax(numNodes, 0);
    for(int i = startTime; i < startTime + windowSize; i++){
        for(int j{0}; j < numNodes; j++){
            //if(nodeMax[j].real() < abs((XValueHistory[j][i]).real())){
            if(sqrt(norm(nodeMax[j])) < abs(sqrt(norm(XValueHistory[j][i])))){
                //nodeMax[j] = abs((XValueHistory[j][i]).real());
                nodeMax[j] = abs(sqrt(norm(XValueHistory[j][i])));
            }
        }
    }
    return nodeMax;
}

double Dynamics::calculateTwoNormVals(std::vector<std::complex<double>> XValueHistory, int startTime, int windowSize){ // end time = numOfWindows*windowSize + startTime
    double nodeMax = 0;
    for(int i = startTime; i < startTime + windowSize; i++){
        //if(nodeMax < abs(XValueHistory[i].real())){
        if(nodeMax < abs(sqrt(norm(XValueHistory[i])))){
            //nodeMax = abs(XValueHistory[i].real());
            nodeMax = abs(sqrt(norm(XValueHistory[i])));
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
