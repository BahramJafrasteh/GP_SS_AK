// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //

//  This file and Control.cpp provide some functions to control the command line
// and preparing and postprocessing data.




#ifndef CONTROL_H
#define CONTROL_H
#include <iostream>
#include <ctime>
#include <cstring>
#include<armadillo>
#include "StreamInt.h"
using namespace arma;
// Command line control class header.
using namespace std;
class Control {
public: 
  // Constructor given the input arguments.
  Control(int argc, char** argv);
  virtual ~Control() {}
  
  bool isArg(string shortName, string longName);
  void UnkFlg();
  
  void readDataFile(mat &X, mat &y, int *data_size, const string fileName);
  int* readDataSize(const string fileName);
  void prepareData(mat& X, mat& y, int &Data_mode, bool &yscale, string ModelN);
  void postData(mat& X, mat& y, bool &yscale, string ModelN);
  void postData(mat& X, bool &yscale, string ModelN);
   void ErrorTermination(const string error);
   void Helping();
  void postData_var(mat& X, bool &yscale, string ModelN);
  void MeanStd(mat& X, mat& y, int &Data_mode, bool &yscale);
  void zeroandone(mat& X, mat& y, int &Data_mode, bool &yscale);
  void prep_symmetric(mat& X, mat& y, int &Data_mode, bool &yscale);
  // exit normally.
  void NormalTermination();
  
  void StatisticsCalc(mat &Xtr, mat&Ytr)
  {
    int inpD = Xtr.n_cols;
    int outD = Ytr.n_cols;
    int NumData = Xtr.n_rows;
    MaxTotalin = Xtr.max();
    MinTotalin = Xtr.min();
    MaxTotalo = Ytr.max();
    MinTotalo = Ytr.min();
    for (int i = 0; i < (inpD + outD); i++)
    {
      if (i == 0)
      {
	
	MinData[i] = MinTotalo;
	MaxData[i] = MaxTotalo;
	MeanData[i] = accu(Ytr)/NumData;
	StData[i] = sqrt( accu(pow(Ytr - MeanData[i], 2)) / (NumData -1) );
      }
      else
      {
	MinData[i] = Xtr.col(i-1).min();
	MaxData[i] = Xtr.col(i-1).max();
	MeanData[i] = accu( Xtr.col(i-1) )/NumData;
	StData[i] = sqrt( accu(pow(Xtr.col(i-1) - MeanData[i], 2)) / (NumData -1) );
      }
    }
  }

 
  
  bool isFlgs() const
  {
    return flgs && getArgNo()<argc;
  }
  void setFlgs(bool val) 
  {
    flgs = val;
  }
  
  // get and set the verbosity level.
  int getVerbose() const
  {
    return verbose;
  }
  void setVerbose(int val)
  {
    verbose = val;
  }
  int getprepM() const
  {
    return prepareM;
  }
  void setprepM(int val)
  {
    prepareM = val;
  }
  
  
  // increase argumentNo by one
  void incArg()
  {
    argNo++;
  }
  int getArgNo() const
  {
    return argNo;
  }
  void setArgNo(int val)
  {
    argNo=val;
  }
  
  string getArg() const
  {
    
    return argv[argNo];
  }
  int getIntArg() const
  {
    return atol(argv[argNo]);
  }
  double getArgLen() const
  {
    return strlen(argv[argNo]);
  }
  // test if the argument is a flag (i.e. starts with -)
  bool isArgFlg() const
  {
    if(argv[argNo][0]=='-')
      return true;
    else
      return false;
  }
  // manipulate the mode --- typically for determining what the help file output will be 
  void setMode(string val) 
  {
    mode = val;
  }
  string getMode() const
  {
    return mode;
  }
  
  mutable mat MinData;
  mutable mat MaxData;
  mutable mat MeanData;
  mutable mat StData;
  mutable double MaxTotalin;
  mutable double MinTotalin;
  mutable double MaxTotalo;
  mutable double MinTotalo;
  mutable mat params;
private:
  bool flgs;
  int verbose;
  int argNo;
  int prepareM;
  string mode;
protected:
  int argc; 
  char** argv; 
  
};


#endif

