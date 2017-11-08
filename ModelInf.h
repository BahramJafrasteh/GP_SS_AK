#ifndef CDATAMODEL_H
#define CDATAMODEL_H
#include <iostream>
#include <string>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>
#include "StreamInt.h"
#include <armadillo>
using namespace arma;

inline double sign(double val)
{
  if (val <= 0)
    return -1;
  else
    return 1;
}
// model inforamtion.
class ModelInfo 
{
public:
  // Initialise the model.
  ModelInfo() {
    _init();
  }
  virtual ~ModelInfo() {}
  ModelInfo(unsigned int nData) : numData(nData) {}
  
  // Get the long name of the model.
  inline string getName() const
  {
    return modelName;
  }
  // Set the long name of the model.
  inline void setName(const string name)
  {
    modelName = name;
  }
  
  // Get the inference type.
  inline string getInf() const
  {
    return InfName;
  }
  // Set the inference type.
  inline void setInf(const string name)
  {
    InfName = name;
  }
  // get likelihood type.
  inline string getlik() const
  {
    return likeName;
  }
  // Set the likelihood type.
  inline void setlik(const string name)
  {
    likeName = name;
  }
  // get mean function type
  inline string getMean() const
  {
    return MeanName;
  }
  // Set the mean function type.
  inline void setMean(const string name)
  {
    MeanName = name;
  }
  // get number of parameters.
  virtual unsigned int getNumPars() const=0;
  virtual void ShowKernelPars(ostream& os) const=0;
  virtual inline unsigned int getNumData() const 
  {
    return numData;
  }
  virtual inline void setNumData(unsigned int val) 
  {
    numData = val;
  }
  inline void ErrorTermination(const string error) 
  { 
    cerr << error << endl << endl;
    exit(1);
  }
private: 
  void _init()
  {
  }
  string modelName;
  string InfName;
  string likeName;
  string MeanName;
  unsigned int numData;
};

// a model which maps from one data space to another.
class Modeling : public ModelInfo
{
public:
  Modeling() : ModelInfo() 
  {
    _init();
  }
  Modeling(unsigned int inDim, unsigned int outDim, unsigned int nData) : inputDim(inDim), outputDim(outDim), ModelInfo(nData) 
  {
    _init();
  }
  // Computing output.
  virtual void Calc_Out(mat& yPred, const mat& inData) const=0;
  
  // Set the input dimension.
  inline void setInpDim(unsigned int dim)
  {
    inputDim = dim;
  }
  // Get the input dimension.
  inline unsigned int getInpDim() const
  {
    return inputDim;
  }
  // Set the output dimension.
  inline void setOutDim(unsigned int dim)
  {
    outputDim = dim;
  }
  // Get the output dimension.
  inline unsigned int getOutDim() const
  {
    return outputDim;
  }
  // Set number of mean function parameters.
  inline void setNumMFpar(unsigned int nmf)
  {
    NumMF = nmf;
  }
  // get number of mean function parameters.
  inline unsigned int getNumMFpar() const
  {
    return NumMF;
  }
  
  // set number of covariance function parameters.
  inline void setNumCovpar(unsigned int covf)
  {
    NumCov = covf;
  }
  // get number of covariance function parameters.
  inline unsigned int getNumCovpar() const
  {
    return NumCov;
  }
  // set number of likelihood function parameters.
  inline void setNumlikfpar(unsigned int likf)
  {
    Numlikf = likf;
  }
  // get number of likelihood function parameters.
  inline unsigned int getNumlikfpar() const
  {
    return Numlikf;
  }
  
  
private:
  void _init()
  {
  }
  unsigned int outputDim;
  unsigned int inputDim;
  unsigned int NumMF;
  unsigned int Numlikf;
  unsigned int NumCov;
  string modelName;
  string type;
};

#endif
