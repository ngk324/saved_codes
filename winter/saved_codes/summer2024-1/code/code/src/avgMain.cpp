
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string.h>
#include <string>


 

int main(int argc, char *argv[]){

    std::string filename = argv[1];


    std::vector<double> twoNormsR;
    std::string line;
    std::ifstream myfile (filename);
    if (myfile.is_open())
    {
        while ( getline (myfile,line) )
        {
            //if(line.length() != 1){
            //if(line == "\n"){
                twoNormsR.push_back(stof(line));
                //std::cout << line << '\n';
           // }
            //else{

            //}
        }
        myfile.close();
    }
    else std::cout << "Unable to open file"; 
    
    double avg = 0;
    for(int i = 0; i < twoNormsR.size(); i++){
        avg += twoNormsR[i];
    }
    avg = avg / ((double)twoNormsR.size());
    std::cout << "Total # of values: " << twoNormsR.size() << "  AVG: " << avg  << std::endl;
    myfile.close();

    return 0;
}


