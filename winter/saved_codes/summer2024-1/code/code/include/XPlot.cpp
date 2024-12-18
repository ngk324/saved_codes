#include "XPlot.hpp"
#include <numeric>
#include <vector>

#include "gdMain.hpp"

namespace XPlot{
    void generateXPlot(Gnuplot &plotX, std::vector<double> &XValues){

        double minVal = *std::min_element(XValues.begin(), XValues.end());
        double maxVal = *std::max_element(XValues.begin(), XValues.end());

        plotX << "reset\n";
        plotX << "min=" << minVal << "\n";
        plotX << "max=" << maxVal << "\n";

        plotX << "set yrange [min:max]\n";
        plotX << "set xrange [0:" << XValues.size()+1 << "]\n";
        plotX << "set xlabel 'Iteration'\n";
        plotX << "set ylabel 'X'\n";

        plotX << "plot '-' u 1:2 with lines lc rgb'red' notitle\n"; 


        plotX << "e\n";

        std::string plotName;

        if(beforeGrad){
            plotName="set output 'Before-" + std::to_string(simNum) +  "-TwoNorm-Plot.png'\n";
        } 
        else{
            plotName= "set output 'After-" + std::to_string(simNum) +  "-TwoNorm-Plot.png'\n";
        }

        plotX << "set terminal png\n";
        //plotX << "set output 'TwoNormPlot.png'\n";
        plotX << plotName;
        plotX << "replot\n";

        plotX.flush();
        usleep(10000);

    }
}