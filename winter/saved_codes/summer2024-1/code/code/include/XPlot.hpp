#ifndef _XPLOT_H
#define _XPLOT_H

#include "gnuplot-iostream.h"
#include <vector>

namespace XPlot{
    void generateXPlot(Gnuplot &plotX, std::vector<double> &XValues);
}

#endif
