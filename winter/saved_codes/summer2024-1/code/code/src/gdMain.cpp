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

#include <boost/numeric/odeint.hpp>

#include <iomanip>


// global variables
int idx;    // variable for node where force is applied
bool beforeGrad;    // variable to tell whether simulation is running on plot before or after gradient descent
int simNum; // simulation number
double randProb;    // probability to use for sampling from cauchy
double h;   // h parameter for cauchy
double frequencyFromUniform; // freq chosen from uniform (eigenvalue)
int numNodes;   // number of nodes 

void write_file_results(std::string print){
    std::string fileName = "AAA-TOTAL-TwoNormResults.txt";
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    log << "\n" << print;
    log.close();
}

void write_two_norm_file_results(double twoNormAvg){
    std::string fileName = "Only-TwoNormResults.txt";
    std::ofstream log;
    log.open(fileName, std::ofstream::app);
    log << std::to_string(twoNormAvg) << std::endl;
    log.close();
}


void oneRun(bool weightConstraint, bool runManyTimes, int evNum){
    std::shared_ptr<Graph> graphInit = std::make_shared<Graph>();
    graphInit->constructSimpleGraph();

    if(beforeGrad){
        std::string filename = "42/LapResultsInit42.txt";
        graphInit->computeMatrices(filename);
        //graphInit->computeEigenValues("EVresultsInit.txt");
    }
    else{
        std::string filename = "42/LapResults42.txt";
        graphInit->computeMatrices(filename);
        //graphInit->computeEigenValues("EVresults.txt");
    }


    int MAX_X = 5000;//numNodes;
    int MAX_Y = 5000;//numNodes;
    int PLOT_SCALE = 1;
    int vPad = 1;
    int hPad = 1;
    double damping{0.001}, stiffness{1}, epsilon{0.1};
    double amp{1};
/*
    std::vector<double> ev;
    for(int i = 0; i < graphInit->eigenValues.size(); i++){
        ev.push_back(graphInit->eigenValues[i]);
    }
    
    frequencyFromUniform = sqrt(ev[evNum]);
    std::cout << "Freq choosen from uniform: " << frequencyFromUniform << std::endl;
    std::string printFreqChosen = "Freq choosen from uniform: " + std::to_string(frequencyFromUniform);
    write_file_results(printFreqChosen);
 */
    // Generate plot
    Plot my_plot("State Plot - Chosen EigVal: " + std::to_string(frequencyFromUniform), PLOT_SCALE, vPad, hPad, MAX_X, MAX_Y);
    my_plot.plotGraphCircle(*graphInit);

    /*if(runManyTimes){
        my_plot.displayPlot(false);
    }
    else{
        my_plot.displayPlot(true);
    }*/

    //my_plot.displayPlot(true);
     
    Force my_force(amp, graphInit->nodes.size());
    //my_force.insertForceElement(idx);
    //my_force.insertForceElement(50);
    //my_force.insertForceElement(49);

    // Simulate dynamics
    int simulationTime{50000};
    //int simulationTime{20};
    //int simulationTime{10};
    int simulationSteps{simulationTime * 100};
    Dynamics my_sim(simulationTime, simulationSteps, damping, stiffness, epsilon, simNum);   
    my_sim.runCentralizedDynamics(*graphInit, my_force, my_plot);                    
    /*if(runManyTimes){
        my_plot.displayPlot(false);
    }
    else{
        my_plot.displayPlot(true);
    }*/
}

void runNTimes(int runs, bool weightConstraint){
    //idx = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
    int evNum = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);;
    randProb = (((double)rand() / (double)RAND_MAX));
    simNum = 1;
    for(int i = 1; i < runs*2 + 1; i++){
        if(i % 2 == 1){            
            std::cout << "\n\n\nSim #: " << simNum << std::endl;
            std::string sim = "\n\n\nSim #: " + std::to_string(simNum);
            write_file_results(sim);
            std::cout << "Before Gradient Descent" << std::endl;
            std::string gradient = "Before Gradient Descent";
            write_file_results(gradient);
        }
        else{
            std::cout << "\nAfter Gradient Descent" << std::endl;
            std::string gradient = "\n\n\nAfter Gradient Descent";
            write_file_results(gradient);
        }
        std::cout << "EV #: " << evNum << std::endl;
        std::string evNumS = "EV #: " + std::to_string(evNum);
        write_file_results(evNumS);
        std::cout << "Node force applied: " << idx << std::endl;
        std::string nodeForceApp = "Node force applied: " + std::to_string(idx);
        write_file_results(nodeForceApp);
        beforeGrad = i % 2;
        oneRun(weightConstraint, true, evNum);
        if(i % 2 == 0){
            simNum++;
            //idx = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
            evNum = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
            randProb = (((double)rand() / (double)RAND_MAX));
        }
    }
}

int main(int argc, char *argv[]){
    bool weightConstraint{true};
    //srand(123);
    if(argc==2){
        numNodes = 173;
        int itr = std::stoi(argv[1]);
        h = 0.1;
        for(int i = itr; i < itr+1; i++){
            std::cout << "Sim Num" << i << std::endl;
            idx = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
            int evNum = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
            randProb = (((double)rand() / (double)RAND_MAX));
            simNum = 1 + (i / 2);
            beforeGrad = (i+1) % 2;
            oneRun(weightConstraint, false, evNum);
        }
    }
    if(argc==3){
        numNodes = std::stoi(argv[1]);
        h = std::stof(argv[2]);
        for(int i = 0; i < numNodes; i++){
            //srand(123);
            srand(1234);
            idx = i;
            runNTimes(100, weightConstraint);
        }
    }
    else if(argc==4){
        numNodes = std::stoi(argv[1]);
        h = std::stof(argv[2]);
        beforeGrad = std::stoi(argv[3]);
        idx = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
        int evNum = (double)rand() * 1.0 / (double)RAND_MAX * (double)(numNodes);
        randProb = (((double)rand() / (double)RAND_MAX));
        std::cout << " EV #: " << evNum << std::endl;
        std::cout << " Node force applied: " << idx << std::endl;
        simNum = 1;
        oneRun(weightConstraint, false, evNum);
    }
    else{
        return 0;
    }
}
