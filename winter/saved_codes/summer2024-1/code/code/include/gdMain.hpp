#ifndef _gdMain
#define _gdMain

// global variables
extern int idx; // variable for node where force is applied
extern bool beforeGrad; // variable to tell whether simulation is running on plot before or after gradient descent
extern int simNum;  // simulation number
extern double randProb; // probability to use for sampling from cauchy
extern double h;    // h parameter for cauchy
extern double frequencyFromUniform;   // freq chosen from uniform (eigenvalue)
extern int numNodes;   // number of nodes 


#endif