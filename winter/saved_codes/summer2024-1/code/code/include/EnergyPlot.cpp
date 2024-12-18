#include "EnergyPlot.hpp"
#include <numeric>
#include <vector>
#include <string>
#include <string.h>

#include "gdMain.hpp"


namespace energyPlot{
    void generateEnergyPlot(Gnuplot &plotEnergy, std::vector<double> &energyValues){

        double minVal = *std::min_element(energyValues.begin(), energyValues.end());
        double maxVal = *std::max_element(energyValues.begin(), energyValues.end());

        plotEnergy << "reset\n";
        plotEnergy << "min=" << minVal << "\n";
        plotEnergy << "max=" << maxVal << "\n";

        plotEnergy << "set yrange [min:max]\n";
        plotEnergy << "set xrange [0:" << energyValues.size()+1 << "]\n";
        plotEnergy << "set xlabel 'Iteration'\n";
        plotEnergy << "set ylabel 'Energy Value'\n";

        plotEnergy << "plot '-' u 1:2 with lines lc rgb'red' notitle\n"; 

        for (size_t i = 0; i < energyValues.size(); ++i) {
            plotEnergy << i << " " << energyValues[i] << "\n";
        }

        plotEnergy << "e\n";

        std::string plotName;

        if(beforeGrad){
            plotName="set output 'Before-" + std::to_string(simNum) +  "-Energy-Plot.png'\n";
        } 
        else{
            plotName= "set output 'After-" + std::to_string(simNum) +  "-Energy-Plot.png'\n";
        }

        plotEnergy << "set terminal png\n";
        //plotEnergy << "set output 'EnergyPlot.png'\n";
        plotEnergy << plotName;
        plotEnergy << "replot\n";

        plotEnergy.flush();
        usleep(10000);
    }
}