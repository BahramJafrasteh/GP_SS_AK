// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
/*This file and Opt_pars contain base class for optimization algorithms.*/

#ifndef Optimization_H
#define Optimization_H

#include <sstream>
#include <stdio.h>
#include <iostream> 
#include <fstream>
#include <string>
#include<armadillo>

using namespace arma; 
using namespace std;

class Opt_Algs {
  
public:
  enum 
  {
    SCG, 
    BFGS,
    LBFGS,
  };
  Opt_Algs()
  {
    // set optimisation parameters
    setVerbose(0);
    setOptimiser(LBFGS);
    setMaxIters(100);
    setTolObjVal(1e-0);
    setTolPars(1e0);
    iter = 0;
    NumfuncEval = 0;
  }
  
  ~Opt_Algs() {}
  virtual inline void setVerbose(int val) const 
  {
    verbose = val;
  }  
  virtual inline int getVerbose() const 
  {
    return verbose;
  }
  virtual unsigned int getNumPars() const=0;
  virtual void get_GP_Pars(mat& param) const=0;
  virtual void set_GP_Pars(mat& param) const=0;
  virtual double Grad_Values(mat& g) const=0;
  virtual double ObjVal() const=0;
  
  void cauchy_point(const  mat g, const mat X,
  const mat Wk, const mat Mk, mat &C, mat &xcp, mat &index_r, const double theta,
  const double mnc
 			  );
  void Primal_Conjugate_grad(const mat index_r,
  const mat xcp, const mat X, const mat Wk, const mat Mk,
  const mat C, const mat g, const double theta, mat &direction
);
  void Zoom( double &steplength_low, double &steplength_high,
		     double &final_steplength, const double f0, const double alpha,
		     const double beta, const mat X0, const mat g0, const mat& direction);
  void LineSearch(const double fold, const mat X, const mat g, double &final_steplength,
			  const mat direction);
  
  void LBFGSOptimise();
  // BFGS algorithm.
  void BFGSOptimize ();
  void Efficient_line_search(
  const double fxk,
  const arma::mat X,
  const mat gk,
  mat &sk,
  double &steplength
);
  void scgOptimise();
  
  double rando()
  {
    return (double)rand() / ((double)RAND_MAX + 1);
  }
  inline int sgn(double val) {
    if (val < 0) return -1;
    if (val >= 0) return 1;
    return 1;
  }
  double ChkBnd (mat &A, const mat lb, const mat ub)
  {
    uvec indl = find(A < lb);
    A.elem(indl) = lb.elem(indl);
    uvec indu = find(A > ub);
    A.elem(indu) = lb.elem(indu);
  }
  bool ChkBndStat (mat &A, const mat lb, const mat ub)
  {
    uvec indl = find(A < lb);
    if (indl.n_elem > 0)
      return true;
    uvec indu = find(A > ub);
    if (indu.n_elem > 0)
      return true;
    return false;
  }  
  
  void setMaxIters(unsigned int val)
  {
    maxIters = val;
  }
  unsigned int getMaxIters() const
  {
    return maxIters;
  }
  void setTolObjVal(double val)
  {
    ToObj = val;
  }
  double getTolObjVal() const
  {
    return ToObj;
  }
  
  
  
  void setTolPars(double val)
  {
    TolPars = val;
  }
  double getTolPars() const
  {
    return TolPars;
  }
  
  void setOptimiser(int val) const
  {
    DefOpt = val;
  }
  int getOptimiser() const
  {
    return DefOpt;
  }
  void setOptimiserStr(string val)
  {
    if(val == "SCG")
      DefOpt = SCG;
    else if(val == "BFGS")
      DefOpt = BFGS;
    else if(val == "LBFGS")
      DefOpt = LBFGS;  
    else 
    {cout <<"Unknown optimisation. \n";
      exit(1);}
  }
  string getDefaultOptimiserStr() const
  {
    switch(DefOpt)
    {
      case SCG:
	return "SCG";
      case BFGS:
	return "quasinew";
      case LBFGS:
	return "LBFGS";
      default:
      {cout <<"Unknown optimisation. \n";
	exit(1);
      }
      
    }
  }
  
  void Optimise()
  {
    switch(DefOpt)
    {
      case SCG:
	scgOptimise();
	break;
      case BFGS:
	BFGSOptimize();
	break;
      case LBFGS:
	LBFGSOptimise();
	break;
      default:
      {cout << "Unknown optimisation.\n";
	exit(1);
      }
      
    }
  }
  
private:
  
  double ToObj;
  double TolPars;
  
  mat lb;
  mat ub;
  
  
  unsigned int iter;
  unsigned int NumfuncEval;
  
  unsigned int maxIters;
  unsigned int maxFuncVal;
  
  mutable int verbose; 
  
  mutable int DefOpt;
  
  bool funcEvalTerminate;
  bool iterTerminate;
  bool fail_pre_bfgs;
  mutable bool been_used_bfgs;
  mutable mat H_bfgs;
  mutable double signalfa_bfgs;
  mutable double last_alpha;
  mutable double min_f_bfgs;
  mutable mat prev_x_bfgs;
  mutable mat prev_direction_bfgs;
  mutable mat prev_derivative_bfgs;
  mutable bool been_used_twice_bfgs;
  mutable mat gamma_bfgs;
  mutable mat delta_bfgs;
  mat direction; // direction for 1-D optimisation.
  mat paramStoreOne;
  mat paramStoreTwo;
}; 
//   double foo(const VectorXd& x, VectorXd& grad);

class Main_Opt_Algs : public Opt_Algs
{
public:
  Main_Opt_Algs() : Opt_Algs() {}
  virtual double logLikelihood() const=0;
  virtual double GradLL(mat& g) const=0;
  virtual double Grad_Values(mat& g) const 
  {
    double L = GradLL(g);
    return L;
  }
  
  virtual double ObjVal() const 
  {
    return logLikelihood();
  }
  
};
inline double put_in_range(const double& a, const double& b, const double& val)
{ 
  
  if (a < b)
  {
    if (val < a)
      return a;
    else if (val > b)
      return b;
  }
  else
  {
    if (val < b)
      return b;
    else if (val > a)
      return a;
  }
  
  return val;
}
inline double poly_min_extrap (
  double f0,
  double d0,
  double f1,
  double d1,
  double limit = 1
)
{
  const double n = 3*(f1 - f0) - 2*d0 - d1;
  const double e = d0 + d1 - 2*(f1 - f0);
  
  
  // find the minimum of the derivative of the polynomial
  
  double temp = max(n*n - 3*e*d0,0.0);
  
  if (temp < 0)
    return 0.5;
  
  temp = sqrt(temp);
  
  if (abs(e) <= numeric_limits<double>::epsilon())
    return 0.5;
  
  // figure out the two possible min values
  double x1 = (temp - n)/(3*e);
  double x2 = -(temp + n)/(3*e);
  
  // compute the value of the interpolating polynomial at these two points
  double y1 = f0 + d0*x1 + n*x1*x1 + e*x1*x1*x1;
  double y2 = f0 + d0*x2 + n*x2*x2 + e*x2*x2*x2;
  
  // pick the best point
  double x;
  if (y1 < y2)
    x = x1;
  else
    x = x2;
  
  // now make sure the minimum is within the allowed range of [0,limit] 
  return put_in_range(0,limit,x);
} 

#endif
