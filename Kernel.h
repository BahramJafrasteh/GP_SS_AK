// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
/* This file and Kernels.cpp contain some classes for representing 
 * kernels.  HybKerns provides a class which forms an 
 * additive kernel from individual kernels 
 * (i.e. a linear plus an RBF kernel).*/


#ifndef KERNELS_H
#define KERNELS_H
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <math.h>
#include "ModelInf.h"
#include<armadillo>

using namespace std;
using namespace arma;


class Kernels : public StreamIntfce {
public:
  Kernels() {}
  Kernels(const mat& X) {}
  Kernels(unsigned int inDim)  {}
  Kernels(const Kernels& kern) {}
  
  virtual ~Kernels()  {}
  virtual Kernels* clone() const=0;
  // set the initial parameters.
  virtual void setInitPars()=0;
  // compute an element of the diagonal.
  virtual double Diag_Kernel(const mat& X, unsigned int index) const=0;
  
  // compute diagonal matrix
  virtual void diag_Compute(mat& d, const mat& X) const
  {
    for(unsigned int i=0; i<X.n_rows; i++)
      d(i) = Diag_Kernel(X, i);
  }
  
  
  // Set parameters of the kernel.
  virtual void setParam(double, unsigned int)=0;
  
  
  // Compute the kernel matrix.
  virtual void computeK(const mat& X1, const mat& X2, mat& K, mat& D2) const = 0;
  // Compute the gradient of the kernel
  virtual void getGradients(mat& g, const mat& X, const mat& X2,  const mat& D2,const mat& QW) const
  {
    getGradients(g, X, X2, D2, QW);
  }
  
  
  // adding NEW KERNEL
  virtual unsigned int addNewKernel(const Kernels* kern)
  {
    cerr << "Error in adding new kernel." << endl;
    return 0;
  }
  // Get a particular parameter of the kernel matrix
  virtual double getParam(unsigned int) const=0;
  
  // Set the parameters of the kernel matrix
  void setParams(const mat& Vector)
  {
    for(unsigned int i=0; i<nParams; i++)
      setParam(Vector(i), i);
  }
  // Get parameters of the kernel.
  void getParams(mat& Vector) const
  {
    for(unsigned int i=0; i<nParams; i++)
      Vector(i) = getParam(i);
  }
  
  // Get the name of kernel.
  inline string getKerName() const
  {
    return kernName;
  }
  // Set the name of kernel.
  inline void setKerName(const string name)
  {
    kernName = name;
  }
  // Set the input dimension.
  inline void setInputDim(unsigned int dim)
  {
    inputDim = dim;
  }
  // Get the input dimension.
  inline unsigned getInputDim() const
  {
    return inputDim;
  }
  // Number of kernel parameters
  inline unsigned int getNPars() const
  {
    return nParams;
  }
  inline unsigned int setNPars(unsigned int np)
  {
    nParams = np;
  }
  // Names of the kernel parameters.
  void setParamName(const string name, unsigned int index)
  {
    
    if(paramNames.size() == index)
      paramNames.push_back(name);
    else 
    {
      if(paramNames.size()<index)
	paramNames.resize(index+1, "no name");
      paramNames[index] = name;
    }
  }
  
  virtual string getParamName(unsigned int index) const
  {
    return paramNames[index];
  }
  
  virtual void ToFile_GP_Params(ostream& out) const;
  
  virtual void FromFile_GP_Params(istream& in);
  
  // Display the kernel.
  virtual ostream& ShowKernelPars(ostream& os) const;
  
  // Get the gradient parameters.
  void GetGrads(mat& g, const mat& X, const mat& X2,  const mat& D2,const mat& QW) const;
  
protected:
  unsigned int nParams;
  string kernName;
  vector<string> paramNames;
private:
  unsigned int inputDim;
};


double EuclDist(mat X1,  mat X2, 
		mat &D2, double hyp);
double MahaDist(mat XX1,  
		mat XX2, mat &D2, mat &ParamKer);
void mlA(mat &AX, const mat X, double hyp);


class mainKernel : public Kernels 
{
public:
  mainKernel() : Kernels() {}
  mainKernel(unsigned int inDim) : Kernels(inDim) {}
  mainKernel(const mat& X) : Kernels(X) {}
  mainKernel(const mainKernel& kern) : Kernels(kern), MainKEl(kern.MainKEl) {}
  virtual unsigned int addNewKernel(const Kernels* kern)
  {
    MainKEl.push_back(kern->clone());
    unsigned int oldNParams = nParams;
    nParams+=kern->getNPars();
    return MainKEl.size()-1;
  }
  virtual void setParam(double val, unsigned int paramNo)
  {
    unsigned int start = 0;
    unsigned int end = 0;
    for(size_t i=0; i<MainKEl.size(); i++)
    {
      end = start+MainKEl[i]->getNPars()-1;
      if(paramNo <= end)
      {
	MainKEl[i]->setParam(val, paramNo-start);
	return;
      }      
      start = end + 1;
    }
  }
  
  virtual double getParam(unsigned int paramNo) const
  {
    unsigned int start = 0;
    unsigned int end = 0;
    for(size_t i=0; i<MainKEl.size(); i++)
    {
      end = start+MainKEl[i]->getNPars()-1;
      if(paramNo <= end)
	return MainKEl[i]->getParam(paramNo-start);
      start = end + 1;
    }
    return -1;
  }
  virtual string getParamName(unsigned int paramNo) const
  {
    unsigned int start = 0;
    unsigned int end = 0;
    for(size_t i=0; i<MainKEl.size(); i++)
    {
      end = start+MainKEl[i]->getNPars()-1;
      if(paramNo <= end)
	return MainKEl[i]->getParamName(paramNo-start);
      start = end + 1;
    }
    return "";
  }
  
  
  virtual void FromFile_GP_Params(istream& in); 
  virtual void ToFile_GP_Params(ostream& out) const;
  virtual unsigned int getNumKerns() const
  {
    return MainKEl.size();
  }
  
  
  
protected:
  vector<Kernels*> MainKEl;
  
};

// Compound Kernel --- This kernel combines other kernels additively together.
class HybKerns: public mainKernel {
public:
  HybKerns();
  HybKerns(unsigned int inDim);
  HybKerns(const mat& X);
  ~HybKerns();
  HybKerns(const HybKerns&);
  HybKerns* clone() const
  {
    return new HybKerns(*this);
  }
  
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index1) const;
  void diag_Compute(mat& d, const mat& X) const;
  
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;
		void getGradients(mat& g, const mat& X, const mat& X2,  const mat& D2,const mat& QW) const;
		
private:
  void _init();
};


// White Noise Kernel.
class Kern_White: public Kernels {
public:
  Kern_White();
  Kern_White(unsigned int inDim);
  Kern_White(const mat& X);
  ~Kern_White();
  Kern_White(const Kern_White&);
  Kern_White* clone() const
  {
    return new Kern_White(*this);
  }
  
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index) const;
  void diag_Compute(mat& d, const mat& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  
  double getGradParam(unsigned int index, const mat& X, 
				const mat& X2,  const mat& D2, const mat& QW) const ;
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;
private:
  void _init();
  double Sigma_White;
};

// Bias Kernel.
class Kern_Bias: public Kernels {
public:
  Kern_Bias();
  Kern_Bias(unsigned int inDim);
  Kern_Bias(const mat& X);
  ~Kern_Bias();
  Kern_Bias(const Kern_Bias&);
  Kern_Bias* clone() const
  {
    return new Kern_Bias(*this);
  }
  
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index) const;
  void diag_Compute(mat& d, const mat& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  
  
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;

void getGradients(mat& g, const mat& X, 
	      const mat& X2, const mat& D2,const mat& QW) const;
		
private:
  void _init();
  double Sigma_Bias;
  
};

// RBF Kernel, also known as the Gaussian or squared exponential kernel.
class Kern_RBF: public Kernels {
public:
  Kern_RBF();
  Kern_RBF(unsigned int inDim);
  Kern_RBF(const mat& X);
  ~Kern_RBF();
  Kern_RBF(const Kern_RBF&);
  Kern_RBF* clone() const
  {
    return new Kern_RBF(*this);
  }
  
  void setInverseWidth(double val)
  {
    inverseWidth_RBF = val;
  }
  double getInverseWidth() const
  {
    return inverseWidth_RBF;
  }
  void setLengthScale(double val)
  {
    inverseWidth_RBF = 1/(val*val);
  }
  double getLengthScale() const
  {
    return 1/sqrt(inverseWidth_RBF);
  }
  void setHayper_Euc(double val)
  {
    Hayper_Euc_RBF = val;
  }
  double getHayper_Euc() const
  {
    return Hayper_Euc_RBF;
  }
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index) const;
  void diag_Compute(mat& d, const mat& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  
  
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;
		
		void getGradients(mat& g, const mat& X, const mat& X2, const mat& D2, const mat& QW) const;
		
		
private:
  void _init();
  double Sigma_RBF;
  double inverseWidth_RBF;
  double Hayper_Euc_RBF;
  
};



// written by Bahram Jafrasteh
class Kern_Exponential: public Kernels {
public:
  Kern_Exponential();
  Kern_Exponential(unsigned int inDim);
  Kern_Exponential(const mat& X);
  ~Kern_Exponential();
  Kern_Exponential(const Kern_Exponential&);
  Kern_Exponential* clone() const
  {
    return new Kern_Exponential(*this);
  }
  
  void setHayper_Euc(double val)
  {
    Hayper_Euc_Exp = val;
  }
  double getHayper_Euc() const
  {
    return Hayper_Euc_Exp;
  }
  
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index) const;
  void diag_Compute(mat& d, const mat& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;
		
		void getGradients(mat& g, const mat& X, const mat& X2, const mat& D2,const mat& QW) const;
		
private:
  void _init();
  double Sigma_Exp;
  double Hayper_Euc_Exp;
   
};

///////////////////////////////////////////////////////////////////
// Exponential Anisotropic Kernel, written by Bahram Jafrasteh
class Kern_ExpAnisotropic: public Kernels {
public:
  Kern_ExpAnisotropic();
  Kern_ExpAnisotropic(unsigned int inDim);
  Kern_ExpAnisotropic(const mat& X);
  ~Kern_ExpAnisotropic();
  Kern_ExpAnisotropic(const Kern_ExpAnisotropic&);
  Kern_ExpAnisotropic* clone() const
  {
    return new Kern_ExpAnisotropic(*this);
  }
  
  void setAngleX(double val)
  {
    AngleX_ExpAns = val;
  }
  double getAngleX() const
  {
    return AngleX_ExpAns;
  }
  void setAngleY(double val)
  {
    AngleY_ExpAns = val;
  }
  double getAngleY() const
  {
    return AngleY_ExpAns;
  }
  void setAngleZ(double val)
  {
    AngleZ_ExpAns = val;
  }
  
  double getAngleZ() const
  {
    return AngleZ_ExpAns;
  }
  double getInverseWidthx() const
  {
    return inverseWidthx_ExpAns;
  }
  void setInverseWidthx(double val)
  {
    inverseWidthx_ExpAns = val;
  }
  double getInverseWidthy() const
  {
    return inverseWidthy_ExpAns;
  }
  void setInverseWidthy(double val)
  {
    inverseWidthy_ExpAns = val;
  }
  double getInverseWidthz() const
  {
    return inverseWidthz_ExpAns;
  }
  void setInverseWidthz(double val)
  {
    inverseWidthz_ExpAns = val;
  }
  void setInitPars();
  double Diag_Kernel(const mat& X, unsigned int index) const;
  void diag_Compute(mat& d, const mat& X) const;
  void setParam(double val, unsigned int paramNum);
  double getParam(unsigned int paramNum) const;
  
  void computeK(const mat& X1, 
		const mat& X2, mat& K, mat& D2) const;
		
		
		void getGradients(mat& g, const mat& X, const mat& X2,  const mat& D2,const  mat& QW) const;
		
private:
  void _init();
  double Sigma_ExpAns;
  double AngleX_ExpAns;
  double inverseWidthx_ExpAns;
  double AngleY_ExpAns;
  double inverseWidthy_ExpAns;
  double AngleZ_ExpAns;
  double inverseWidthz_ExpAns;
  double InversewidthR_ExpAns;
};



void WriteKernelPas(const Kernels& kern, ostream& out);
Kernels* ReadKerFromFile(istream& in);


#endif
