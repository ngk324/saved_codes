#ifndef _TWONORMPLOT_H
#define _TWONORMPLOT_H

#include "gnuplot-iostream.h"
#include <vector>

namespace twoNormPlot{
    void generateTwoNormPlot(Gnuplot &plotTwoNorm, std::vector<std::complex<double>> &twoNormValues);
}

#endif
