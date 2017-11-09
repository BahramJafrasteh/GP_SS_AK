// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
#include "Kernel.h"

using namespace std;

ostream& Kernels::ShowKernelPars(ostream& os) const
{
  os << getKerName() << " kernel:" << endl;
  for(unsigned int i=0; i<nParams; i++)
  { 
    os << getParamName(i) << ": " << getParam(i) << endl;
  }
  return os;
}
// Writing GP Kernels parameters.
void Kernels::ToFile_GP_Params(ostream& out) const
{
  out << "KernelName="<<getKerName() << endl;
  out << "inputDim="<<getInputDim() << endl;
  out << "numParams="<<getNPars() << endl;
  mat par(1, getNPars());
  getParams(par);
  for(unsigned int i = 0; i < par.n_rows; i++) 
  {
    for(unsigned int j = 0; j < par.n_cols; j++) 
    {
      double val = par(i, j);
      if((val - (int)val)==0.0)
	out << (int)val << " ";
      else
	out << val << " ";
    }
    out << endl;
  }

}




void Kernels::GetGrads(mat& g, const mat& X, const mat& X2, const mat &D2, const mat& QW) const
{

  getGradients(g, X, X2, D2, QW);


}


// The main kernel
void mainKernel::FromFile_GP_Params(istream& in) 
{

  unsigned int NumberOfKernels = ReadIntStrm(in, "NumberOfKernels");
  for(unsigned int i=0; i<NumberOfKernels; i++)
      addNewKernel(ReadKerFromFile(in));

  
}
// Getting Information from Kernels
void mainKernel::ToFile_GP_Params(ostream& out) const
{
  
  out << "KernelName="<<getKerName() << endl;
  out << "NumberOfKernels="<<getNumKerns() << endl;
  for(unsigned int i=0; i<MainKEl.size(); i++)
  {
    MainKEl[i]->StrmOut(out);
  }
  
}

// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
// // // // // // // // // // // The Hybrid kernel.  // // // // // // // // // // // // 
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
// // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // // 
HybKerns::HybKerns() : mainKernel()
{
  _init();
}
HybKerns::HybKerns(unsigned int inDim) : mainKernel(inDim)
{
  _init();
  setInputDim(inDim);
}
HybKerns::HybKerns(const mat& X) : mainKernel(X)
{
  _init();
  setInputDim(X.n_cols);
}  
HybKerns::HybKerns(const HybKerns& kern) : mainKernel(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  for(size_t i=0; i<MainKEl.size(); i++)
    addNewKernel(MainKEl[i]->clone()); 
}

HybKerns::~HybKerns()
{
  for(size_t i=0; i<MainKEl.size(); i++)
    delete MainKEl[i];
}
void HybKerns::_init()
{
  nParams=0;
  setKerName("Hyb");
  setInitPars();
}

void HybKerns::setInitPars()
{
}
double HybKerns::Diag_Kernel(const mat& X, unsigned int index) const
{
  double y=0.0;
  for(size_t i=0; i<MainKEl.size(); i++)
    y+=MainKEl[i]->Diag_Kernel(X, index);
  return y;
}

void HybKerns::diag_Compute(mat& d, const mat& X) const
{
  d.zeros();
  mat dStore = zeros<mat>(d.n_rows, d.n_cols);
  for(size_t i=0; i < MainKEl.size(); i++)
  {
    MainKEl[i]->diag_Compute(dStore, X);
    d += dStore;
  }
}



void HybKerns::computeK(const mat& X1, 
				  const mat& X2, mat& K, mat& D2) const
{
  double y=0.0;
  mat K_t = zeros<mat>(K.n_rows, K.n_cols);
  mat D2_t = zeros<mat>(D2.n_rows, D2.n_cols);
  D2.zeros();
  K.zeros();
  for(size_t i=0; i<MainKEl.size(); i++)
  {
    MainKEl[i]->computeK(X1, X2, K_t, D2_t);
    D2 += D2_t;
    K += K_t;
  }
}
  
  void HybKerns::getGradients(mat& g, const mat& X, const mat& X2, const mat& D2,const mat& QW) const
{
  unsigned int start = 0;
  unsigned int end = 0;
  for(size_t i=0; i<MainKEl.size(); i++)
  {
    end = start+MainKEl[i]->getNPars()-1;
    mat subg(1, MainKEl[i]->getNPars());
    MainKEl[i]->getGradients(subg, X, X2, D2, QW);

    g.submat(0, start, 0, end) = subg;
    start = end+1;
  }
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////White noise Kernel /////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
 
Kern_White::Kern_White() : Kernels()
{
  _init();
}
Kern_White::Kern_White(unsigned int inDim) : Kernels(inDim)
{
  _init();
  setInputDim(inDim);
}
Kern_White::Kern_White(const mat& X) : Kernels(X)
{
  _init();
  setInputDim(X.n_cols);
}  
Kern_White::Kern_White(const Kern_White& kern) : Kernels(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  Sigma_White = kern.Sigma_White;
}

Kern_White::~Kern_White()
{
}

void Kern_White::_init()
{
  nParams = 1;
  setKerName("White Noise");
  setParamName("Sigma_White", 0);
  setInitPars();

}
// Changing this parameter could affect the performanc of the GP.
void Kern_White::setInitPars()
{
  Sigma_White = 0.10;  
}

inline double Kern_White::Diag_Kernel(const mat& X, unsigned int index) const
{
  return Sigma_White;
}

void Kern_White::diag_Compute(mat& d, const mat& X) const
{
  d.fill(Sigma_White);
}

void Kern_White::setParam(double val, unsigned int paramNo)
{
  switch(paramNo)
  {
  case 0:
    Sigma_White = val;
    break;
  default:
  {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}

double Kern_White::getParam(unsigned int paramNo) const
{
  switch(paramNo)
  {
  case 0:
    return Sigma_White;
    break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}


inline void Kern_White::computeK(const mat& X1, 
				  const mat& X2, mat& K, mat& D2) const
{
D2.zeros();
K.zeros();
if (X1(0) == X2(0) && X1.n_rows == X2.n_rows)
  K.diag().fill(Sigma_White);
}

double Kern_White::getGradParam(unsigned int index, const mat& X, 
				const mat& X2,  const mat& D2, const mat& QW) const 
{
  return 0.0;
}





/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////Bias Kernel /////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////

Kern_Bias::Kern_Bias() : Kernels()
{
  _init();
}
Kern_Bias::Kern_Bias(unsigned int inDim) : Kernels(inDim)
{
  _init();
  setInputDim(inDim);
}
Kern_Bias::Kern_Bias(const mat& X) : Kernels(X)
{
  _init();
  setInputDim(X.n_cols);
}  
Kern_Bias::Kern_Bias(const Kern_Bias& kern) : Kernels(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  Sigma_Bias = kern.Sigma_Bias;
  
}
Kern_Bias::~Kern_Bias()
{
}

void Kern_Bias::_init()
{
  nParams = 1;
  setKerName("Bias");
  setParamName("Sigma_Bias", 0);
  setInitPars();

}
// Changing this parameter could affect the performanc of the GP.
void Kern_Bias::setInitPars()
{
  Sigma_Bias = 0.2;  
}

double Kern_Bias::Diag_Kernel(const mat& X, unsigned int index) const
{
  return Sigma_Bias;
}


void Kern_Bias::diag_Compute(mat& d, const mat& X) const
{

  d.fill(Sigma_Bias);
}

void Kern_Bias::setParam(double val, unsigned int paramNo)
{
  switch(paramNo)
  {
  case 0:
    Sigma_Bias = val;
    break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}
double Kern_Bias::getParam(unsigned int paramNo) const
{
  switch(paramNo)
  {
  case 0:
    return Sigma_Bias;
    break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}




inline void Kern_Bias::computeK(const mat& X1, 
				  const mat& X2, mat& K, mat& D2) const
{
  D2.zeros();
  K.fill(Sigma_Bias);
}


void Kern_Bias::getGradients(mat& g, const mat& X, const mat& X2, const mat& D2,const mat& QW) const
{
  uvec indices = regspace<uvec>(0,  QW.n_elem - 1);
  mat RColon = QW.elem(indices);
  mat R = eye(QW.n_rows, QW.n_rows);
  mat D2Colon = R.elem(indices);
  g[0] = accu( RColon.t() * D2Colon);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // // // // // // // // // // // // // // // // RBF kernel. // // // // // // // // // // // //
////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////
Kern_RBF::Kern_RBF() : Kernels()
{
  _init();
}
Kern_RBF::Kern_RBF(unsigned int inDim) : Kernels(inDim)
{
  _init();
  setInputDim(inDim);
}
Kern_RBF::Kern_RBF(const mat& X) : Kernels(X)
{
  _init();
  setInputDim(X.n_cols);
}  
Kern_RBF::Kern_RBF(const Kern_RBF& kern) : Kernels(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  Sigma_RBF = kern.Sigma_RBF;
  inverseWidth_RBF = kern.inverseWidth_RBF;
  Hayper_Euc_RBF = kern.Hayper_Euc_RBF;
}

Kern_RBF::~Kern_RBF()
{
}

void Kern_RBF::_init()
{
  nParams = 3;
  setKerName("RBF");
  setParamName("Hayper_Euc_RBF", 0);

  setParamName("inverseWidth_RBF", 1);

  setParamName("Sigma_RBF", 2);
  setInitPars();

}
// Changing these parameters could affect the performanc of the GP.
void Kern_RBF::setInitPars()
{
  Hayper_Euc_RBF = 0.5;//0.01;
  inverseWidth_RBF = 0.9;//;
  Sigma_RBF = 0.5;//0.1
  
}

inline double Kern_RBF::Diag_Kernel(const mat& X, unsigned int index) const
{
  return Sigma_RBF*Sigma_RBF;
}

void Kern_RBF::diag_Compute(mat& d, const mat& X) const
{
  d.fill(Sigma_RBF*Sigma_RBF);
}

void Kern_RBF::setParam(double val, unsigned int paramNo)
{
  switch(paramNo)
  {
  case 0:
    Hayper_Euc_RBF = val;
    break;
  case 1:
    inverseWidth_RBF = val;
    break;
  case 2:
    Sigma_RBF = val;
    break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}
double Kern_RBF::getParam(unsigned int paramNo) const
{
  switch(paramNo)
  {
    case 0:
      return Hayper_Euc_RBF;
      break;
  case 1:
    return inverseWidth_RBF;
    break;
  case 2:
    return Sigma_RBF;
    break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}




void Kern_RBF::computeK(const mat& X1, 
				  const mat& X2, mat& K, mat& D2) const
{
  double var2 = Sigma_RBF * Sigma_RBF;  
  EuclDist(X1,X2, D2, Hayper_Euc_RBF);
    K = exp(-0.5*inverseWidth_RBF * D2) * var2;
}


void Kern_RBF::getGradients(mat& g, const mat& X, const mat& X2, const mat& D2,const mat& QW) const
{
  double g1=0.0;
  double g2=0.0;
  double g3=0.0;
  mat D2Colon(X.n_rows*X2.n_rows,1);
  mat KD2(X.n_rows, X2.n_rows);
  mat dk(X.n_rows, X2.n_rows);
  mat dk_tmp(X.n_rows, X2.n_rows);
  mat R(X.n_rows, X2.n_rows);
  mat RColon(X.n_rows*X2.n_rows, 1);
  mat Qt(X.n_rows*X2.n_rows, 1);
  mat Q(X.n_rows,X2.n_rows);
  mat Qs(X.n_rows,X2.n_rows);
  mat Qx(X.n_rows, 1);
  mat Qx2(X2.n_rows, 1);
  mat dhp(1,1);
  mat Di2(D2.n_rows, D2.n_cols);
  mat Qwidth(X.n_rows, X2.n_rows);
  double var2 = Sigma_RBF * Sigma_RBF;
  
  Q = QW;
  Qs = var2 * QW;
   KD2 = exp ( -0.5*inverseWidth_RBF * D2 );

  dk = exp ( -inverseWidth_RBF/2 * D2) * (-inverseWidth_RBF/2);
  R = Qs % dk;
  uvec indices = regspace<uvec>(0,  R.n_elem - 1);
  RColon = R.elem(indices);
  D2Colon = D2.elem(indices);
  dhp = -2.0 * RColon.t() * D2Colon;
  g1 = dhp(0);

  Qwidth = -0.5 * (Qs % KD2) % D2;
  g2 = accu(Qwidth);

  Q %= KD2;
  Qt = Q.t();  
  Qt *= Sigma_RBF;
  Q *= Sigma_RBF;
  Qx = sum(Q, 0);
  Qx2 = sum(Qt, 0);
  g3 = accu(Qx) + accu(Qx2);
  g3 = g3 * Sigma_RBF;
  
  // finally 
  g(0) = g1/2;
  g(1) = g2/2;
  g(2) = g3/2;

}





//////////////////////////////////////////////////////////////////////////////////////////////
// 		    Exponential Kernel					   //
//////////////////////////////////////////////////////////////////////////////////////////////
Kern_Exponential::Kern_Exponential() : Kernels()
{
  _init();
}
Kern_Exponential::Kern_Exponential(unsigned int inDim) : Kernels(inDim) {
  _init();
  setInputDim(inDim);
}
Kern_Exponential::Kern_Exponential(const mat& X) : Kernels(X)
{
  _init();
  setInputDim(X.n_cols);
}

Kern_Exponential::Kern_Exponential(const Kern_Exponential& kern) : Kernels(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  Sigma_Exp = kern.Sigma_Exp;
  Hayper_Euc_Exp = kern.Hayper_Euc_Exp;
}

Kern_Exponential::~Kern_Exponential()
{
}

void Kern_Exponential::_init()
{
  nParams = 2;
  setKerName("Exp");
  setParamName("Hayper_Euc_Exp", 0);
  setParamName("Sigma_Exp", 1);
  setInitPars();
  
}
// Changing these parameters could affect the performanc of the GP.
void Kern_Exponential::setInitPars()
{
  Hayper_Euc_Exp = 0.5;
  Sigma_Exp = 0.9;
}

inline double Kern_Exponential::Diag_Kernel(const mat& X, unsigned int index) const
{
  return Sigma_Exp*Sigma_Exp;
}

void Kern_Exponential::diag_Compute(mat& d, const mat& X) const
{
  d.fill(Sigma_Exp*Sigma_Exp);
}

void Kern_Exponential::setParam(double val, unsigned int paramNo)
{
  switch(paramNo)
  {
    case 0:
      Hayper_Euc_Exp = (val);
      break;
    case 1:
      Sigma_Exp = (val);
      break;
    default:
    {cout <<"Requested parameter doesn't exist.\n";
      exit(1);}
  }
}
double Kern_Exponential::getParam(unsigned int paramNo) const
{
  switch(paramNo)
  {
    case 0:
      return (Hayper_Euc_Exp);
      break;
    case 1:
      return (Sigma_Exp);
      break;
    default:
    {cout <<"Requested parameter doesn't exist.\n";
      exit(1);}
  }
}




void Kern_Exponential::computeK(const mat& X1, const mat& X2, mat& K, mat& D2) const
{
  double var2 = Sigma_Exp * Sigma_Exp;  
  EuclDist(X1,X2, D2, Hayper_Euc_Exp);
  K = var2 * exp(-1 * sqrt(D2));
  
}


// gradient of the kernel function
void Kern_Exponential::getGradients(mat& g, const mat& X, const mat& X2,  const mat& D2,const mat& QW) const
{

  mat D2Colon(X.n_rows*X2.n_rows,1);
  mat KD2(X.n_rows, X2.n_rows);
  mat dk(X.n_rows, X2.n_rows);
  mat dk_tmp(X.n_rows, X2.n_rows);
  mat R(X.n_rows, X2.n_rows);
  mat RColon(X.n_rows*X2.n_rows, 1);
  mat Qt(X.n_rows*X2.n_rows, 1);
  mat Q(X.n_rows,X2.n_rows);
  mat Qs(X.n_rows,X2.n_rows);
  mat Qx(X.n_rows, 1);
  mat Qx2(X2.n_rows, 1);
  mat dhp(1,1);
  mat Di2(D2.n_rows, D2.n_cols);
  double var2 = Sigma_Exp * Sigma_Exp;
  Q = QW;
  Qs = var2*QW ;
  
  KD2 = exp(-1 * sqrt(D2) );
  
  dk = sqrt(D2);
  
  dk_tmp = -0.5/dk;
  dk = exp(-1* dk) % dk_tmp;
  dk.diag().fill(0);
  
  R = Qs % dk;
  
  uvec indices = regspace<uvec>(0,  R.n_elem-1);
  RColon = R.elem(indices);
  
  Di2 = D2;
  D2Colon = Di2.elem(indices);
  
  dhp = RColon.t() * D2Colon;
  g[0] = dhp(0);
  
  Q %= KD2;
  

    ///////////////// grad of  Sigma //////////////////////////
   D2Colon = Q.elem(indices);
   RColon = KD2.elem(indices);
   dhp = RColon.t() * D2Colon;
   g[1] = dhp(0) * Sigma_Exp;

  
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Exponential Anisotropic Kernel
////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////
Kern_ExpAnisotropic::Kern_ExpAnisotropic() : Kernels()
{
  _init();
}
Kern_ExpAnisotropic::Kern_ExpAnisotropic(unsigned int inDim) : Kernels(inDim) {
  _init();
  setInputDim(inDim);
}
Kern_ExpAnisotropic::Kern_ExpAnisotropic(const mat& X) : Kernels(X)
{
  _init();
  setInputDim(X.n_cols);
}

Kern_ExpAnisotropic::Kern_ExpAnisotropic(const Kern_ExpAnisotropic& kern) : Kernels(kern)
{
  _init();
  setInputDim(kern.getInputDim());
  AngleX_ExpAns = kern.AngleX_ExpAns;
  inverseWidthx_ExpAns = kern.inverseWidthx_ExpAns;
  AngleY_ExpAns = kern.AngleY_ExpAns;
  inverseWidthy_ExpAns = kern.inverseWidthy_ExpAns;
  AngleZ_ExpAns = kern.AngleZ_ExpAns;
  inverseWidthz_ExpAns = kern.inverseWidthz_ExpAns;
  Sigma_ExpAns = kern.Sigma_ExpAns;
  InversewidthR_ExpAns = kern.InversewidthR_ExpAns;
  
}

Kern_ExpAnisotropic::~Kern_ExpAnisotropic()
{
}

void Kern_ExpAnisotropic::_init()
{
  nParams = 8;
  setKerName("ExpAns");
  setParamName("AngleX_ExpAns", 0);

  setParamName("inverseWidthx_ExpAns", 1);

 
  setParamName("AngleY_ExpAns", 2);

  setParamName("inverseWidthy_ExpAns", 3);

 
  setParamName("AngleZ_ExpAns", 4);

  setParamName("inverseWidthz_ExpAns", 5);

 
  setParamName("Sigma_ExpAns", 6);
  
  setParamName("InversewidthR_ExpAns", 7);
  setInitPars();

}
// Changing these parameters could affect the performanc of the GP.
void Kern_ExpAnisotropic::setInitPars()
{
  AngleX_ExpAns = M_PI/3.1;
inverseWidthx_ExpAns = 1.5;//0.6
  AngleY_ExpAns = M_PI/3.1;
  inverseWidthy_ExpAns = 1.5;//1.1
  AngleZ_ExpAns = M_PI/3.1;
  inverseWidthz_ExpAns = 1.3;//0.65
  Sigma_ExpAns = 0.9;//0.9
  InversewidthR_ExpAns = 0.6;  
}

inline double Kern_ExpAnisotropic::Diag_Kernel(const mat& X, unsigned int index) const
{
  return Sigma_ExpAns*Sigma_ExpAns;
}

void Kern_ExpAnisotropic::diag_Compute(mat& d, const mat& X) const
{
  d.fill(Sigma_ExpAns*Sigma_ExpAns);
}

void Kern_ExpAnisotropic::setParam(double val, unsigned int paramNo)
{
  switch(paramNo)
  {
    case 0:
      AngleX_ExpAns = val;
      break;
    case 1:
      inverseWidthx_ExpAns = (val);
      break;
    case 2:
      AngleY_ExpAns = val;
      break;
    case 3:
      inverseWidthy_ExpAns = val;
      break;
    case 4:
            AngleZ_ExpAns = val;
      break;
    case 5:
      inverseWidthz_ExpAns = val;
      break;
    case 6:
      Sigma_ExpAns = val;
      break;
    case 7:
      InversewidthR_ExpAns = (val);
      break;
    default:
      {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}
double Kern_ExpAnisotropic::getParam(unsigned int paramNo) const
{
  switch(paramNo)
  {
    case 0:
      return AngleX_ExpAns;
      break;
   case 1:
      return (inverseWidthx_ExpAns);
      break;
    case 2:
      return AngleY_ExpAns;
      break;
    case 3:
      return (inverseWidthy_ExpAns);
      break;
    case 4:
      return AngleZ_ExpAns;
      break;
    case 5:
      return (inverseWidthz_ExpAns);
      break;
    case 6:
      return (Sigma_ExpAns);
      break;
    case 7:
      return (InversewidthR_ExpAns);
      break;
  default:
    {cout <<"Requested parameter doesn't exist.\n";
    exit(1);}
  }
}





void Kern_ExpAnisotropic::computeK(const mat& X1, 
				  const mat& X2, mat& K, mat& D2) const
{

  
  double var2 = Sigma_ExpAns * Sigma_ExpAns;  
  

  mat ParamKer;
  if (X1.n_cols == 3)
  {
    ParamKer.resize(1, 6);
    ParamKer = {AngleX_ExpAns, AngleY_ExpAns, AngleZ_ExpAns,
      inverseWidthx_ExpAns, inverseWidthy_ExpAns, inverseWidthz_ExpAns};
      
  }
  else if (X1.n_cols == 4)
  {
     ParamKer.resize(1, 7);
    ParamKer = {AngleX_ExpAns, AngleY_ExpAns, AngleZ_ExpAns,
      inverseWidthx_ExpAns, inverseWidthy_ExpAns, 
      inverseWidthz_ExpAns, InversewidthR_ExpAns};
  }
   MahaDist(X1, X2, D2, ParamKer);
  
  K = var2 * exp(-1 * sqrt(D2));
}



void Kern_ExpAnisotropic::getGradients(mat& g, const mat& X, const mat& X2,  const mat& D2,const mat& QW) const
{

  mat D2Colon= zeros<mat>(X.n_rows*X2.n_rows,1);
  mat KD2= zeros<mat>(X.n_rows, X2.n_rows);
  mat dk= zeros<mat>(X.n_rows, X2.n_rows);
  mat dk_tmp= zeros<mat>(X.n_rows, X2.n_rows);
  mat R= zeros<mat>(X.n_rows, X2.n_rows);
  mat RColon= zeros<mat>(X.n_rows*X2.n_rows, 1);
  mat Qt= zeros<mat>(X.n_rows*X2.n_rows, 1);
  mat Q= zeros<mat>(X.n_rows,X2.n_rows);
  mat Qs= zeros<mat>(X.n_rows,X2.n_rows);
  mat Qx= zeros<mat>(X.n_rows, 1);
  mat Qx2= zeros<mat>(X2.n_rows, 1);
  mat dhp= zeros<mat>(1,1);
  mat Di2(D2.n_rows, D2.n_cols);
  mat SD2(D2.n_rows, D2.n_cols);
  mat DX(D2.n_rows, D2.n_cols);
  mat DY(D2.n_rows, D2.n_cols);
  mat DZ(D2.n_rows, D2.n_cols);
  ////////////////////////
  mat DD2(SD2.n_rows, SD2.n_cols);
  double var2 = Sigma_ExpAns * Sigma_ExpAns;
  double ncls = X.n_cols;
  mat ParamKer;
    if (X.n_cols == 3)
  {
    ParamKer.resize(1,6);
    ParamKer = {AngleX_ExpAns, AngleY_ExpAns, AngleZ_ExpAns,
      inverseWidthx_ExpAns, inverseWidthy_ExpAns, 
      inverseWidthz_ExpAns};
  }
  else if (X.n_cols == 4)
  {
    ParamKer.resize(1,7);
    ParamKer = {AngleX_ExpAns, AngleY_ExpAns, AngleZ_ExpAns,
      inverseWidthx_ExpAns, inverseWidthy_ExpAns, 
      inverseWidthz_ExpAns, InversewidthR_ExpAns};    
  }
     MahaDist(X, X2, DD2, ParamKer);
  Q = QW;
  Qs = var2 * QW;
  double alpha = AngleX_ExpAns;
  double beta = AngleY_ExpAns;
  double teta = AngleZ_ExpAns;
  mat Rot(ncls, ncls);
  Rot.zeros();
  mat Rot_alpha(ncls, ncls);
  Rot_alpha.zeros();
  mat Rot_beta(ncls, ncls);
  Rot_beta.zeros();
  mat Rot_teta(ncls, ncls);
  Rot_teta.zeros();
  mat S(ncls, ncls);
  S.zeros();
  mat S_alpha(ncls, ncls);
  S_alpha.zeros();
  mat S_beta(ncls, ncls);
  S_beta.zeros();
  mat S_teta(ncls, ncls);
  S_teta.zeros();
  mat S_Lalpha(ncls, ncls);
  S_Lalpha.zeros();
  mat S_Lbeta(ncls, ncls);
  S_Lbeta.zeros();
  mat S_Lteta(ncls, ncls);
  S_Lteta.zeros();
  mat InversewidthRock(ncls, ncls);
  InversewidthRock.zeros();
  Rot (0, 0) = cos(alpha) * cos(teta) + sin(alpha) * sin(beta) * sin(teta);
  Rot_alpha (0, 0) = - sin(alpha) * cos(teta) + cos(alpha) * sin(beta) * sin(teta);
  Rot_beta (0, 0) =  sin(alpha) * cos(beta) * sin(teta);
  Rot_teta (0, 0) = -cos(alpha) * sin(teta) + sin(alpha) * sin(beta) * cos(teta);
  
  Rot (0, 1) = -sin(alpha) * cos(teta) + cos(alpha) * sin(beta) * sin(teta);
  Rot_alpha (0, 1) = -cos(alpha) * cos(teta) - sin(alpha) * sin(beta) * sin(teta);
  Rot_beta (0, 1) = cos(alpha) * cos(beta) * sin(teta);
  Rot_teta (0, 1) = sin(alpha) * sin(teta) + cos(alpha) * sin(beta) * cos(teta);
  
  Rot (0, 2) = -cos(beta) * sin(teta);
  Rot_alpha (0, 2) = 0.0;
  Rot_beta (0, 2) = sin(beta) * sin(teta);
  Rot_teta (0, 2) = -cos(beta) * cos(teta);
  
  Rot (1, 0) = sin(alpha) * cos(beta);
  Rot_alpha (1, 0) = cos(alpha) * cos(beta);
  Rot_beta (1, 0) = -sin(alpha) * sin(beta);
  Rot_teta(1, 0) = 0.0;
  
  Rot (1, 1) = cos(alpha) * cos(beta);
  Rot_alpha (1, 1) = -sin(alpha) * cos(beta);
  Rot_beta (1, 1) = -cos(alpha) * sin(beta);
  Rot_teta (1, 1) = 0.0;
  
  Rot (1, 2) = sin(beta);
  Rot_alpha (1, 2) = 0.0;
  Rot_beta (1, 2) = cos(beta);
  Rot_teta (1, 2) = 0.0;
  
  Rot (2, 0) = cos(alpha) * sin(teta) - sin(alpha) * sin(beta) * cos(teta);
  Rot_alpha (2, 0) = -sin(alpha) * sin(teta) - cos(alpha) * sin(beta) * cos(teta);
  Rot_beta (2, 0) = - sin(alpha) * cos(beta) * cos(teta);
  Rot_teta (2, 0) = cos(alpha) * cos(teta) + sin(alpha) * sin(beta) * sin(teta);
  
  Rot (2, 1) = -sin(alpha) * sin(teta) - cos(alpha) * sin(beta) * cos(teta);
  Rot_alpha (2, 1) = -cos(alpha) * sin(teta) + sin(alpha) * sin(beta) * cos(teta);
  Rot_beta (2, 1) = - cos(alpha) * cos(beta) * cos(teta);
  Rot_teta (2, 1) = -sin(alpha) * cos(teta) + cos(alpha) * sin(beta) * sin(teta);
  
  Rot (2, 2) = cos(beta) * cos(teta);
  Rot_alpha (2, 2) = 0.0;
  Rot_beta (2, 2) = -sin(beta) * cos(teta);
  Rot_teta (2, 2) = -cos(beta) * sin(teta);
  
  S(0, 0) = inverseWidthx_ExpAns * pow (Rot(0, 0), 2) + 
  inverseWidthy_ExpAns * pow (Rot(0, 1), 2) +
  inverseWidthz_ExpAns * pow (Rot(0, 2), 2);
  S_alpha(0, 0) = inverseWidthx_ExpAns * 2 * Rot(0, 0) * Rot_alpha(0, 0) + 
  inverseWidthy_ExpAns * 2 * Rot(0, 1) * Rot_alpha(0, 1) +
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_alpha(0, 2);
  S_beta(0, 0) = inverseWidthx_ExpAns * 2 * Rot(0, 0) * Rot_beta(0, 0) + 
  inverseWidthy_ExpAns * 2 * Rot(0, 1) * Rot_beta(0, 1) +
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_beta(0, 2);
  S_teta(0, 0) = inverseWidthx_ExpAns * 2 * Rot(0, 0) * Rot_teta(0, 0) + 
  inverseWidthy_ExpAns * 2 * Rot(0, 1) * Rot_teta(0, 1) +
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_teta(0, 2);
  S_Lalpha(0, 0) = pow (Rot(0, 0), 2);
  S_Lbeta(0, 0) = pow (Rot(0, 1), 2);
  S_Lteta(0, 0) = pow (Rot(0, 2), 2);
  
  S(0, 1) = inverseWidthx_ExpAns * Rot(0, 0) * Rot(1, 0) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot(1, 1) +
  inverseWidthz_ExpAns * Rot(0, 2) * Rot(1, 2);   
  S_alpha(0, 1) = inverseWidthx_ExpAns * Rot_alpha(0, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(0, 0) * Rot_alpha(1, 0)
  + inverseWidthy_ExpAns * Rot_alpha(0, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot_alpha(1, 1)+
  inverseWidthz_ExpAns * Rot_alpha(0, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_alpha(1, 2);
  S_beta(0, 1) = inverseWidthx_ExpAns * Rot_beta(0, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(0, 0) * Rot_beta(1, 0)
  + inverseWidthy_ExpAns * Rot_beta(0, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot_beta(1, 1)+
  inverseWidthz_ExpAns * Rot_beta(0, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_beta(1, 2);
  S_teta(0, 1) = inverseWidthx_ExpAns * Rot_teta(0, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(0, 0) * Rot_teta(1, 0)
  + inverseWidthy_ExpAns * Rot_teta(0, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot_teta(1, 1)+
  inverseWidthz_ExpAns * Rot_teta(0, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_teta(1, 2);
  S_Lalpha(0, 1) =Rot(0, 0) * Rot(1, 0) ;
  S_Lbeta(0, 1) =  Rot(0, 1) * Rot(1, 1);
  S_Lteta(0, 1) =  Rot(0, 2) * Rot(1, 2);    
  
  S(0, 2) = inverseWidthx_ExpAns * Rot(0, 0) * Rot(2, 0) + inverseWidthy_ExpAns * Rot(0, 1) * Rot(2, 1) +
  inverseWidthz_ExpAns * Rot(0, 2) * Rot(2, 2);
  S_alpha(0, 2) = inverseWidthx_ExpAns * Rot_alpha(0, 0) * Rot(2, 0) + inverseWidthx_ExpAns * Rot(0, 0) * Rot_alpha(2, 0) +
  inverseWidthy_ExpAns * Rot_alpha(0, 1) * Rot(2, 1) + inverseWidthy_ExpAns * Rot(0, 1) * Rot_alpha(2, 1) +
  inverseWidthz_ExpAns * Rot_alpha(0, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_alpha(2, 2);   
  S_beta(0, 2) = inverseWidthx_ExpAns * Rot_beta(0, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(0, 0) * Rot_beta(2, 0) +
  inverseWidthy_ExpAns * Rot_beta(0, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot_beta(2, 1) +
  inverseWidthz_ExpAns * Rot_beta(0, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_beta(2, 2);
  S_teta(0, 2) = inverseWidthx_ExpAns * Rot_teta(0, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(0, 0) * Rot_teta(2, 0) +
  inverseWidthy_ExpAns * Rot_teta(0, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(0, 1) * Rot_teta(2, 1) +
  inverseWidthz_ExpAns * Rot_teta(0, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(0, 2) * Rot_teta(2, 2);    
  S_Lalpha(0, 2) = Rot(0, 0) * Rot(2, 0);
  S_Lbeta(0, 2) = Rot(0, 1) * Rot(2, 1);
  S_Lteta(0, 2) = Rot(0, 2) * Rot(2, 2);
  
  S(1, 0) = S(0, 1);
  S_alpha(1, 0) = S_alpha(0, 1);
  S_beta(1, 0) = S_beta(0, 1);
  S_teta(1, 0) = S_teta(0, 1);
  S_Lalpha(1, 0) = S_Lalpha(0, 1);
  S_Lbeta(1, 0) = S_Lbeta(0, 1);
  S_Lteta(1, 0) = S_Lteta(0, 1);
  
  
  S(1, 1) = inverseWidthx_ExpAns * Rot(1, 0) * Rot(1, 0) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot(1, 1) +
  inverseWidthz_ExpAns * Rot(1, 2) * Rot(1, 2);
  S_alpha(1, 1) = inverseWidthx_ExpAns * Rot_alpha(1, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_alpha(1, 0)+
  inverseWidthy_ExpAns * Rot_alpha(1, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_alpha(1, 1) +
  inverseWidthz_ExpAns * Rot_alpha(1, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_alpha(1, 2);
  S_beta(1, 1) = inverseWidthx_ExpAns * Rot_beta(1, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_beta(1, 0)+
  inverseWidthy_ExpAns * Rot_beta(1, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_beta(1, 1) +
  inverseWidthz_ExpAns * Rot_beta(1, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_beta(1, 2);
  S_teta(1, 1) = inverseWidthx_ExpAns * Rot_teta(1, 0) * Rot(1, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_teta(1, 0)+
  inverseWidthy_ExpAns * Rot_teta(1, 1) * Rot(1, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_teta(1, 1) +
  inverseWidthz_ExpAns * Rot_teta(1, 2) * Rot(1, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_teta(1, 2);
  S_Lalpha(1, 1) = Rot(1, 0) * Rot(1, 0);
  S_Lbeta(1, 1) = Rot(1, 1) * Rot(1, 1);
  S_Lteta(1, 1) = Rot(1, 2) * Rot(1, 2);
  
  
  S(1, 2) = inverseWidthx_ExpAns * Rot(1, 0) * Rot(2, 0) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot(2, 1) +
  inverseWidthz_ExpAns * Rot(1, 2) * Rot(2, 2);
  S_alpha(1, 2) = inverseWidthx_ExpAns * Rot_alpha(1, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_alpha(2, 0) +
  inverseWidthy_ExpAns * Rot_alpha(1, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_alpha(2, 1) +
  inverseWidthz_ExpAns * Rot_alpha(1, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_alpha(2, 2);
  S_beta(1, 2) = inverseWidthx_ExpAns * Rot_beta(1, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_beta(2, 0) +
  inverseWidthy_ExpAns * Rot_beta(1, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_beta(2, 1) +
  inverseWidthz_ExpAns * Rot_beta(1, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_beta(2, 2);
  S_teta(1, 2) = inverseWidthx_ExpAns * Rot_teta(1, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(1, 0) * Rot_teta(2, 0) +
  inverseWidthy_ExpAns * Rot_teta(1, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(1, 1) * Rot_teta(2, 1) +
  inverseWidthz_ExpAns * Rot_teta(1, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(1, 2) * Rot_teta(2, 2);
  S_Lalpha(1, 2) = Rot(1, 0) * Rot(2, 0);
  S_Lbeta(1, 2) = Rot(1, 1) * Rot(2, 1);
  S_Lteta(1, 2) = Rot(1, 2) * Rot(2, 2);    
  
  
  S(2, 0) = S(0, 2);
  S_alpha(2, 0) = S_alpha(0, 2);
  S_beta(2, 0) = S_beta(0, 2);
  S_teta(2, 0) = S_teta(0, 2);
  S_Lalpha(2, 0) = S_Lalpha(0, 2);
  S_Lbeta(2, 0) = S_Lbeta(0, 2);
  S_Lteta(2, 0) = S_Lteta(0, 2);
  
  
  
  S(2, 1) = S(1, 2);
  S_alpha(2, 1) = S_alpha(1, 2);
  S_beta(2, 1) = S_beta(1, 2);
  S_teta(2, 1) = S_teta(1, 2);
  S_Lalpha(2, 1) = S_Lalpha(1, 2);
  S_Lbeta(2, 1) = S_Lbeta(1, 2);
  S_Lteta(2, 1) = S_Lteta(1, 2);
  
  
  S(2, 2) = inverseWidthx_ExpAns * Rot(2, 0) * Rot(2, 0) + 
  inverseWidthy_ExpAns * Rot(2, 1) * Rot(2, 1) +
  inverseWidthz_ExpAns * Rot(2, 2) * Rot(2, 2);
  S_alpha(2, 2) = inverseWidthx_ExpAns * Rot_alpha(2, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(2, 0) * Rot_alpha(2, 0) +
  inverseWidthy_ExpAns * Rot_alpha(2, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(2, 1) * Rot_alpha(2, 1) +
  inverseWidthz_ExpAns * Rot_alpha(2, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(2, 2) * Rot_alpha(2, 2);
  S_beta(2, 2) = inverseWidthx_ExpAns * Rot_beta(2, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(2, 0) * Rot_beta(2, 0) +
  inverseWidthy_ExpAns * Rot_beta(2, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(2, 1) * Rot_beta(2, 1) +
  inverseWidthz_ExpAns * Rot_beta(2, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(2, 2) * Rot_beta(2, 2);
  S_teta(2, 2) = inverseWidthx_ExpAns * Rot_teta(2, 0) * Rot(2, 0) + 
  inverseWidthx_ExpAns * Rot(2, 0) * Rot_teta(2, 0) +
  inverseWidthy_ExpAns * Rot_teta(2, 1) * Rot(2, 1) + 
  inverseWidthy_ExpAns * Rot(2, 1) * Rot_teta(2, 1) +
  inverseWidthz_ExpAns * Rot_teta(2, 2) * Rot(2, 2) + 
  inverseWidthz_ExpAns * Rot(2, 2) * Rot_teta(2, 2);
  S_Lalpha(2, 2) = Rot(2, 0) * Rot(2, 0);
  S_Lbeta(2, 2) = Rot(2, 1) * Rot(2, 1);
  S_Lteta(2, 2) = Rot(2, 2) * Rot(2, 2);
    
  // gradient respect to InversewidthR
  if (ncls > 3)
  {
    InversewidthRock(3, 3) = 1;
    S(3, 3) = 1;
  }
  
  
  KD2 = exp(-1 * sqrt(DD2));
  
  SD2 = sqrt(DD2);
  uvec indi = find(SD2 == 0);
  dk_tmp = -0.5 / SD2;
  dk_tmp.elem(indi).fill(0);
  
  dk = exp(-1.0 * SD2) % dk_tmp;
  dk.diag().zeros();
  R = Qs % dk;
  uvec indices = regspace<uvec>(0,  R.n_elem - 1);
  RColon = R.elem(indices);
         
  mat oness = ones<mat>(1, X2.n_rows);
  mat oness2 = ones<mat>(1, X.n_rows);
  
  Di2  = sum( 2.0*(X % X) * (S % S_alpha) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_alpha) , 1)).t() -
  4.0 * ((X * (S % S_alpha) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[0] = dhp(0);
  
  Di2  = sum( 2.0*(X % X) * (S % S_Lalpha) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_Lalpha) , 1)).t() -
  4.0 * ((X * (S % S_Lalpha) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[1] = dhp(0);
  
  Di2  = sum( 2.0*(X % X) * (S % S_beta) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_beta) , 1)).t() -
  4.0 * ((X * (S % S_beta) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[2] = dhp(0);
  
  Di2  = sum( 2.0*(X % X) * (S % S_Lbeta) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_Lbeta) , 1)).t() -
  4.0 * ((X * (S % S_Lbeta) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[3] = dhp(0);
  
  Di2  = sum( 2.0*(X % X) * (S % S_teta) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_teta) , 1)).t() -
  4.0 * ((X * (S % S_teta) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[4] = dhp(0);
  
  
  Di2  = sum( 2.0*(X % X) * (S % S_Lteta) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % S_Lteta) , 1)).t() -
  4.0 * ((X * (S % S_Lteta) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = RColon.t() * D2Colon;
  g[5] = dhp(0);
  


  
  ///////////////// grad of  Sigma //////////////////////////
   D2Colon = Q.elem(indices);
   RColon = KD2.elem(indices);
   dhp = 2.0 * RColon.t() * D2Colon;
   g[6] = dhp(0) * Sigma_ExpAns;
  
  
  
  if (ncls == 4)
  {
    Di2  = sum( 2.0*(X % X) * (S % InversewidthRock) , 1)* oness + 
  oness2.t() *(sum( 2.0*(X2 % X2) * (S % InversewidthRock) , 1)).t() -
  4.0 * ((X * (S % InversewidthRock) )* X2.t()) ;
  D2Colon = Di2.elem(indices);
  dhp = -2.0 * RColon.t() * D2Colon;
  double g8 = dhp(0);
  g[7] = g8 / (D2.n_rows);
  }
  else if (ncls  == 3)
    g[7] = 0.0;
  
  
  // finally 
//   g = -g;
  
}






// reading and writing the kernel parameters.

void WriteKernelPas(const Kernels& kern, ostream& out)
{
  kern.StrmOut(out);
}





Kernels* ReadKerFromFile(istream& in)
{ 
  Kernels* KernW;
  string line;
  getline(in, line);
  size_t pos = line.find("="); 
  string  KernelN = line.substr (pos+1);
  if(KernelN=="white")
    KernW = new Kern_White();
  else if(KernelN=="Bias")
    KernW = new Kern_Bias();
  else if(KernelN=="RBF")
    KernW = new Kern_RBF();
  else if(KernelN=="Exp")
    KernW = new Kern_Exponential(); 
  else if(KernelN=="ExpAns")
    KernW = new Kern_ExpAnisotropic();    
  else if(KernelN=="Hyb")
    KernW = new HybKerns();
  else
  {cout  << "Unknown kernel type \n";
    exit(1);}
    KernW->FromFile_GP_Params(in);
    
  
  return KernW;
}


void Kernels::FromFile_GP_Params(istream& in)
{ 
  
  setInputDim(ReadIntStrm(in, "inputDim"));
  unsigned int nPars = ReadIntStrm(in, "numParams");
  
  string lineC;
  mat par = zeros<mat>(1, nPars);
    if(!getline(in, lineC))
    {cout << "Can not read " << getKerName() <<" kernel parameters. \n";
      exit(1);}
      
      for (int i = 0; i < nPars; i++)
      {
	size_t pos = lineC.find(" "); 
	string  val = lineC.substr (0, pos+1);
	if (lineC.size() <= 0)
	{
	  cout << "The nember of Hyper-parameters of " << getKerName() 
	  <<" are not sufficient. \n";
	  exit(1);
	}
	lineC.erase(0, pos+1);
	par[i] = atof(val.c_str());
      }
      
      setParams(par);
      
}


///////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////
double EuclDist(mat X1,  
		mat X2, mat &D2, double hyp)
{
  int n = X1.n_rows;
  int m = X2.n_rows;
  mat mX1 = zeros<mat>(1,X1.n_cols);
  mat mX2 = zeros<mat>(1,X2.n_cols);
  mat AX1 = zeros<mat>(X1.n_rows, X1.n_cols);
  mat AX2 = zeros<mat>(X2.n_rows, X2.n_cols);
  
  
  mX1 = (double)n/(n+m) * sum(X1, 0) / X1.n_rows;
  mX2 = (double)m/(n+m) * sum(X2, 0) / X2.n_rows + mX1;
  for (int j = 0; j < X1.n_cols; j++)
  {
    X1.col(j) -= mX2[j];
    X2.col(j) -= mX2[j];
  }
  mlA(AX1, X1, hyp);
  mlA(AX2, X2, hyp);
  mat oness = ones<mat>(1, X2.n_rows);
  mat oness2 = ones<mat>(1, X1.n_rows);
  D2  = sum( AX1 % X1 , 1)* oness + oness2.t() *(sum( AX2 % X2 , 1)).t() - 2.0 * X1* AX2.t();
  uvec indexes= find(D2 < 0);
  D2.elem(indexes).fill(0.0);
}

double MahaDist(mat X1,  
		mat X2, mat &D2, mat &ParamKer)
// 				 double alpha, double beta, double teta,
// 				 double L_alpha, double L_beta, double L_teta)
{
  double alpha = ParamKer[0];
  double beta = ParamKer[1];
  double teta = ParamKer[2];
  double L_alpha = ParamKer[3];
  double L_beta = ParamKer[4];
  double L_teta = ParamKer[5];
  double L_r;
  if (X1.n_cols == 4)
  {
    L_r = ParamKer[6];
  }
  int n = X1.n_rows;
  int m = X2.n_rows;
  int ncls = X1.n_cols;
  mat mX1(1,X1.n_cols);
  mat mX2(1,X2.n_cols);
  mX1 = (double)n/(n+m) * sum(X1, 0) / X1.n_rows;
  mX2 = (double)m/(n+m) * sum(X2, 0) / X2.n_rows + mX1;
  for (int j = 0; j < X1.n_cols; j++)
  {
    X1.col(j) -= mX2[j];
    X2.col(j) -= mX2[j];
  }
  
  mat Rot(ncls, ncls);
  mat lambda (ncls, ncls);
  Rot.zeros();
  Rot (0, 0) = cos(alpha) * cos(teta) + sin(alpha) * sin(beta) * sin(teta);
  Rot (0, 1) = -sin(alpha) * cos(teta) + cos(alpha) * sin(beta) * sin(teta);
  Rot (0, 2) = -cos(beta) * sin(teta);
  Rot (1, 0) = sin(alpha) * cos(beta);
  Rot (1, 1) = cos(alpha) * cos(beta);
  Rot (1, 2) = sin(beta);
  Rot (2, 0) = cos(alpha) * sin(teta) - sin(alpha) * sin(beta) * cos(teta);
  Rot (2, 1) = -sin(alpha) * sin(teta) - cos(alpha) * sin(beta) * cos(teta);
  Rot (2, 2) = cos(beta) * cos(teta);
  if (ncls == 4)
  {
    Rot(3, 3) = 1;
  }
  
  //      Rot = Rot.t();
  lambda.zeros();
  lambda(0, 0) = L_alpha;
  lambda(1, 1) = L_beta;
  lambda(2, 2) = L_teta;
  if (ncls == 4)
  {
    lambda(3, 3) = L_r;
  }
  mat sigInv = (Rot* lambda * Rot.t());
  X1 *= sigInv;
  X2 *= sigInv;
  mat oness = ones<mat>(1, X2.n_rows);
  mat oness2 = ones<mat>(1, X1.n_rows);
  
  D2  = sum( X1 % X1 , 1)* oness 
  + oness2.t() *(sum( X2 % X2 , 1)).t() - 2.0 * X1* X2.t();      
  uvec indexes= find(D2 < 0);
  D2.elem(indexes).fill(0.0);
}

void mlA(mat &AX,  mat X, double hyp)
{
  AX = X;
  AX *= exp((double)-2 * log(hyp));
}

  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
