// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
// This file and GP_utils provide main GP functions.
#ifndef GP_utils_H
#define GP_utils_H
#include <limits>
#include "ModelInf.h"
#include "Opt_pars.h"
#include "StreamInt.h"
#include "Kernel.h"
#include "Opt_pars.h"
#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif 

using namespace std;



class GP_utils : public Modeling, public Main_Opt_Algs, public StreamIntfce
{
public:
  enum likelihoodType
  {
    likeL_Gaussian,
    likeL_WarpGauss
  };
  enum InferenceType
  {
    inf_laplace,
    inf_EP
  };
  enum MeanType
  {
    mean_zero,
    mean_sum
  };
  enum WarpFunc
  {
    tanh1,
    rbf,
    srbf
  };
  GP_utils();
  
  GP_utils(Kernels* kernel, mat Xin, mat Yin, int Inf_type = inf_laplace, int likeLtype = likeL_Gaussian, int mean_type = mean_zero, 
	 unsigned int numhyper = 1,unsigned int numlik_par = 1, unsigned int numMF_par = 0,int verbos=2);
  
  
  // initialize variables
  void initialize_vars();
  
  
  // Modeling
  void Calc_Out(mat& yPred, const mat& inData) const;
  void Calc_Out(mat& yPred, mat& yVar, const mat& inData) const;
  void Calc_Out(mat& yPred, mat& yVar, mat& probPred, const mat& inData) const;
  
  // updateing alpha.
  void updateAlpha() const;
  void PSI(mat &psi, mat &Alp, mat &fval) const;
  void irls() const;
  void brentmin(mat &Fv, mat &dalpha, double &fmin)  const;
  void mvmK_exact(mat alp) const;
  double solve_chol(const mat Lc, mat &Xr, const mat dB) const;
  void updatelikelihood(mat fval) const;
  void updatelikelihood() const;
  void warpingfunction() const;
  void warpingfunction(const mat &yy, mat &gyy, mat &lgpyy) const;
  void inverse_warpingfunction(const mat &Z, mat &G) const;
  void ldB2_exact() const;
  void ldB2_exact_drive(mat dWs);
  void ldB2_exact(mat WW, mat r, mat &QQ) const;
  void updateGlikelihood() const;
  void posteriorMeanVar(mat& mu, mat& varSigma, const mat& X) const;
  void posteriorMean(mat& mu, const mat& X) const;
  void Gauher(mat &weight, vec& abscissas, double N) const;
  
  // updating Kernel parameters.
  void updateKernel() const;
  
  // updating the gradient.
  void updateG() const;
  
  // compute log likelihood.
  virtual double logLikelihood() const;
  // compute the gradients of the log likelihood.
  virtual double GradLL(mat& g) const;
  void dhyp(mat &dhat) const;
  double mvmK_exact( mat X, mat Z, const mat K) const;
  void updateMean() const;
  void updateMean(mat &mfval) const;
  void updateGMean() const;
  
  void OptimisePars(unsigned int iters=1000);
  
  void ShowKernelPars(ostream& os) const;
  
  virtual unsigned int getNumPars() const;
  virtual void get_GP_Pars(mat& param) const;
  virtual void set_GP_Pars(mat& param) const; // override;
  void FromFile_GP_Params(istream& in);
  void ToFile_GP_Params(ostream& out) const;
  
  
  
  
  double getHypermfVal(unsigned int index) const {
    return hypermf( index);
  }
  void setHypermfVal(double val, unsigned int index) {
    hypermf(index) = val;
  }
  void setHypermf(const mat& scal) {
    hypermf = scal;
  } 
  
  double getHyperlfVal(unsigned int index) const {
    return hyperlf( index);
  }
  void setHyperlfVal(double val, unsigned int index) {
    hyperlf(index) = val;
    setLikelihoodUpStat(false);
  }
  void setHyperlf(const mat& scal) {
    
    hyperlf = scal;
    setLikelihoodUpStat(false);
  } 
  
  
  int getLikelihoodType() const {
    return likelihoodType;
  }
  string getLiklihoodStr() const {
    switch(likelihoodType) {
      case likeL_Gaussian:
	return "likeL_Gaussian";
      case likeL_WarpGauss:
	return "likeL_WarpGauss";
      default:
      {cout <<"Unknown approximation type \n";
	exit(1);
      }
    }
  }
  
  void setLikelihoodStr(const string val) {
    if(val=="likeL_Gaussian") 
      setLikelihoodType(likeL_Gaussian);
    else if(val=="likeL_WarpGauss")
      setLikelihoodType(likeL_WarpGauss);
    else
    {cout <<"Unknown approximation type \n";
      exit(1);
    }
  }
  void setLikelihoodType (const int val)
  {
    likelihoodType = val;
    if (likelihoodType == likeL_WarpGauss)
      setWarpfunc(rbf);
    else if (likelihoodType == likeL_Gaussian)
    {
      hyperlf.zeros(1,1);
      g_hyperlf.zeros(1,1);
    }
    
  }
  int getInferenceType() const {
    return InferenceType;
  }
  string getInferenceStr() const {
    switch(InferenceType) {
      case inf_laplace:
	return "inf_laplace";
      case inf_EP:
	return "inf_EP";
      default:
      {cout <<"Unknown approximation type \n";
	exit(1);
      }
    }
  }
  
  void setInferenceStr(const string val) {
    if(val=="inf_laplace") 
      setInferenceType(inf_laplace);
    else if(val=="inf_EP")
      setInferenceType(inf_EP);
    else
    {cout <<"Unknown approximation type \n";
      exit(1);
    }
  }
  void setInferenceType (const int val)
  {
    InferenceType = val;
  }
  
  int getMeanType() const {
    return MeanType;
  }
  string getMeanTypeStr() const {
    switch(MeanType) {
      case mean_zero:
	return "mean_zero";
      case mean_sum:
	return "mean_sum";
      default:
      {cout <<"Unknown approximation type \n";
	exit(1);
      }
    }
  }
  
  void setMeanTypeStr(const string val) {
    if(val=="mean_zero") 
      setMeanType(mean_zero);
    else if(val=="inf_EP")
      setMeanType(inf_EP);
    else
    {cout <<"Unknown approximation type \n";
      exit(1);
    }
  }
  void setMeanType (const int val)
  {
    MeanType = val;
  }
  
  int getWarpfunc() const
  {
    return WarpFunc;
  }
  string getWarpfuncstr() const
  {
    switch (WarpFunc)
    {
      case tanh1:
	return "tanh1";
      case rbf:
	return "rbf";
      case srbf:
	return "srbf";
    }
  }
  void setWarpfunc (const int val)
  {
    WarpFunc = val;
  }
  
  
  // Flag which indicates K needs to be updated.
  bool isUpdateK() const 
  {
    return KUpdateStat;
  }
  void setKUpdateStat(const bool val) const 
  {
    KUpdateStat = val;
    if(!KUpdateStat)
    {
      setAlphaUpStat(false);
    }
  }
  
  bool AlphaUpStat() const 
  {
    return AlphaUpStatus;
  }
  void setAlphaUpStat(const bool val) const 
  {
    AlphaUpStatus = val;
  }
  
  bool LikelihoodUpStat() const 
  {
    return LikelihoodUpStatus;
  }
  void setLikelihoodUpStat(const bool val) const 
  {
    LikelihoodUpStatus = val;
  }
  bool getLikelihoodUpStat() const 
  {
    return(LikelihoodUpStatus);
  }  
  void setlogLikelihoodUpStat(const bool val) const 
  {
    logLikelihoodUpStatus = val;
  }
  bool getlogLikelihoodUpStat() const 
  {
    return(logLikelihoodUpStatus);
  }  
  
  inline const Kernels* getKernel() const {
    return KerenlW;
  }
  
  
  mat Xinp;
  mat yTarg;  // target data.
  
  

  mutable mat dW;
  mutable mat D2;
  mutable mat gy;
  mutable mat lgpy;
  mutable mat L;
  
  mutable mat Alpha;
  mutable mat mvmK;
  mutable mat yhat;
  mutable mat Lchol;
  
  mutable double Lchol_db2;
  mutable mat Sw;
  mutable mat Q;
  mutable mat QW;
  mutable mat hypermf;
  mutable mat hyperlf;  
  mutable mat lp;
  mutable mat dlp;
  mutable mat d2lp;
  mutable mat d3lp;
  mutable mat lp_dhyp;
  mutable mat dlp_dhyp;
  mutable mat d2lp_dhyp;
  mutable mat mf;
  mutable mat dmf;
  mat g;
  
  Kernels* KerenlW;
  
  mutable mat K;
  mutable mat KD2;
  mutable mat invK;
  mutable mat diagK;
  mutable mat LcholK;
  mutable double logDetK;
  
  
  mutable mat A;
  mutable mat Ainv;
  mutable mat LcholA;
  mutable double logDetA;
  
  
  // gradie matrices.
  mutable mat g_hyperlf;
  mutable mat g_hypermf;
  mutable mat g_param; 
  
private:
  void _init();
  void _updateKernel() const; // updating K
  void _ComputeK_NewData(mat& kX, const mat& Xin) const;  
  void _ComputeDiag_NewData(mat &kD, const mat& Xin) const;
  void _postMean(mat& mu, const mat& kX) const; 
  void _postVar(mat& varSigma, mat& kX, const mat& Xin) const;
  
  
  int likelihoodType; // likeL_Gaussian, likeL_WarpGauss
  int InferenceType; 
  int MeanType;
  int WarpFunc;
  
  mutable bool Chol_fail;
  mutable bool KUpdateStat;
  mutable bool AlphaUpStatus;
  mutable bool LikelihoodUpStatus; 
  mutable bool logLikelihoodUpStatus;
  string type;
};

// Functions which operate on the object
void writeGpToStream(const GP_utils& model, ostream& out);
void writeGPFile(const GP_utils& model, const string modelFileName, const string comment="");
GP_utils* readGpFromStream(istream& in);
GP_utils* readGpFromFile(const string modelfileName, int verbosity=2);


#endif /* GP_utils_H */
