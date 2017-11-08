// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
#ifndef GP_SS_AK_H
#define GP_SS_AK_H
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <stdio.h>
#include "Kernel.h"
#include "GP_Utils.h"
#include "Opt_pars.h"
#include "Control.h"
#include <armadillo>
using namespace std;
int main(int argc, char* argv[]);

class GP_Cntrl : public Control {
public: 
  GP_Cntrl(int argc, char** argv);
  void train();
  void test();
  void Help();
};

#else /* GP_SS_AK_H */
#endif /* GP_SS_AK_H */
