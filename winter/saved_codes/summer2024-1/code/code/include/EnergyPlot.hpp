#ifndef _ENERGYPLOT_H
#define _ENERGYPLOT_H

#include "gnuplot-iostream.h"
#include <vector>

namespace energyPlot{
    void generateEnergyPlot(Gnuplot &plotEnergy, std::vector<double> &energyValues);
}

#endif
