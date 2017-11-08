  #include "GP_Utils.h"
  GP_utils::GP_utils()
  : Modeling(), Main_Opt_Algs()
  {
    _init();
  }
  
  
  GP_utils::GP_utils(Kernels* KerenlWel,
		 mat pXin, mat Yin, int Inf_type, int likelihoodType, int mean_type, 
		 unsigned int numhyper, unsigned int numlik_par, 
		 unsigned int numMF_par,int verbos)
  :
  Modeling(), Main_Opt_Algs(),
  KerenlW(KerenlWel), Xinp(pXin), yTarg(Yin)
  {
    
    setNumMFpar(numMF_par);
    setNumlikfpar(numlik_par);
    setNumCovpar(numhyper);
    
    _init();
    setInferenceType(Inf_type);
    setLikelihoodType(likelihoodType);
    setMeanType(mean_type);
    setVerbose(verbos);
    g_param.resize(1, KerenlW->getNPars());
    setOutDim(yTarg.n_cols);
    setInpDim(pXin.n_cols);
    setNumData(yTarg.n_rows);
    initialize_vars();
    
    if (numMF_par > 0)
    {
      hypermf.resize(numMF_par,1);
      for (int i = 0; i < numMF_par; i++)
	hypermf(i) = 0.12;
    }
    if (numlik_par > 0)
    {
      hyperlf.resize(numlik_par,1);
      for (int i = 0; i < numlik_par; i++)
	hyperlf(i) = 0.016; 
    }
    
  }
  void GP_utils::_init()
  {
    setName("Gaussian process");
    setInf("Lapalce");
    setlik("Gaussian");
    setMean("Zero");
    KUpdateStat = false;
    Chol_fail = false;
    
  }
  

  
  void GP_utils::initialize_vars()
  {
    setLikelihoodUpStat(false);   
    setlogLikelihoodUpStat(false);
    K.resize(getNumData(), getNumData());
    D2.resize(getNumData(), getNumData());
    KD2.resize(getNumData(), getNumData());
    invK.resize(getNumData(), getNumData());
    LcholK.resize(getNumData(), getNumData());
    Alpha.resize(getNumData(), getOutDim());
    lp.resize(getNumData(), getOutDim());
    dlp.resize(getNumData(), getOutDim());
    d2lp.resize(getNumData(), getOutDim());
    Sw.resize(getNumData(), getOutDim());
    d3lp.resize(getNumData(), getOutDim());
    lp_dhyp.resize(getNumData(), getOutDim());
    dlp_dhyp.resize(getNumData(), getOutDim());
    d2lp_dhyp.resize(getNumData(), getOutDim());    
    mvmK.resize(getNumData(), getOutDim());    
    mf.resize(getNumData(), getOutDim());  
    dmf.resize(getNumData(), getOutDim());  
    yhat.resize(getNumData(), getOutDim());  
    dW.resize(getNumData(), getOutDim());  
    Lchol.resize(getNumData(), getNumData());
    Q.resize(getNumData(), getNumData());
    QW.resize(getNumData(), getNumData()); 
  }
  
  
  
  
  
  
  
  
  unsigned int GP_utils::getNumPars() const
  {
    int tot = KerenlW->getNPars() + getNumMFpar() + getNumlikfpar();
    return tot;
  }
  
  void GP_utils::get_GP_Pars(mat& param) const
  {
    
    int counter = 0;
    
    for (unsigned int i = 0; i < KerenlW->getNPars(); i++)
    {
      param(counter) =KerenlW->getParam(i);   
      counter++;
    }
    if (getNumlikfpar() > 0)
    {
      for (unsigned int i = 0; i < getNumlikfpar(); i++)
      {
	param(counter) = getHyperlfVal(i);
	counter++;
      }
    }
    if (getNumMFpar() > 0)
    {
      for (unsigned int i = 0; i < getNumMFpar(); i++)
      {
	param(counter) = getHypermfVal(i);
	counter++;
      }
    }
    
  }
  
  void GP_utils::set_GP_Pars(mat& param) const
  {
    setKUpdateStat(false);
    setlogLikelihoodUpStat(false);
    int counter = 0;
    for (unsigned int i = 0; i < KerenlW->getNPars(); i++)
    {
      KerenlW->setParam(param(counter), i);
      counter++;
    }
    if (getNumlikfpar() > 0)
    {
      for (unsigned int i = 0; i < getNumlikfpar(); i++)
      {
	
	hyperlf(i) = param(counter);
	counter++;
      }
    }
    if (getNumMFpar() > 0)
    {
      for (unsigned int i = 0; i < getNumMFpar(); i++)
      {
	hypermf(i) = param(counter);
	counter++;
      }
    }  
  }
  
  void GP_utils::Calc_Out(mat& yPred, const mat& Xin) const
  {
    mat muTest(yPred.n_rows, yPred.n_cols);
    mat varSigmaTest(yPred.n_rows, yPred.n_cols);
    posteriorMeanVar(muTest, varSigmaTest, Xin);
    yPred = muTest;
  }
  void GP_utils::Calc_Out(mat& yPred, mat& yVar, const mat& Xin) const
  {
    mat muTest(yPred.n_rows, yPred.n_cols);
    posteriorMeanVar(muTest, yVar, Xin);
    yPred = muTest;
  }
  
  void GP_utils::Calc_Out(mat& yPred, mat& probPred, mat& yVar, const mat& Xin) const
  {
    mat muTest(yPred.n_rows, yPred.n_cols);
    posteriorMeanVar(muTest, yVar, Xin);
    yPred = muTest;
  }
  
  void GP_utils::PSI(mat &psi, mat &Alp, mat &fval) const
  {
    mat fval_tmp(fval.n_rows, fval.n_cols);
    mvmK_exact(Alp);
    fval = mvmK;
    fval_tmp = fval;
    fval += mf;
    updatelikelihood(fval);
    fval_tmp *= 0.5;
    psi = Alp.t() * fval_tmp - accu(lp);
  }
  void GP_utils::irls() const
  {
    mat Fv = zeros<mat>(getNumData(),1);
    mat psi_new = zeros<mat>(1,1);
    mat psi_old = zeros<mat>(1,1);
    mat W_old = zeros<mat>(d2lp.n_rows, d2lp.n_cols);
    mat dalpha(getNumData(), 1);
    mat B(getNumData(), 1);
    int maxit = 20;
    double Wmin = 0.0;
    double tol = 1e-6;
    double fmin;
    PSI(psi_new, Alpha, Fv);
    psi_old(0) = std::numeric_limits<double>::infinity();
    int it = 0;
    while( (psi_old(0) - psi_new(0)) > tol && it<maxit )
    {
      psi_old(0) = psi_new(0);
      it ++;
      W_old = d2lp;
      uvec ind = find(W_old < Wmin);
      if (accu(ind) > 0)
	W_old.rows(ind).fill(Wmin);
      Fv -= mf;
      B = Fv;
      B = B % d2lp + dlp;
      mvmK_exact(B);
      ldB2_exact(d2lp, mvmK, dalpha);
      if (Chol_fail)
      {
	return;
      }
      dalpha = -1*dalpha - Alpha + B;
      fmin = 0;
      brentmin(Fv, dalpha, fmin);
      psi_new(0) = fmin;
    }
  }
  void GP_utils::brentmin(mat &Fv, mat &dalpha, double &fmin) const
  {
    mat fa = zeros<mat>(1,1);
    mat fb = zeros<mat>(1,1);
    mat fc = zeros<mat>(1,1);
    mat fu = zeros<mat>(1,1);
    mat Xa = zeros<mat>(getNumData(),1);
    mat Xb = zeros<mat>(getNumData(),1);
    mat Xc = zeros<mat>(getNumData(),1);
    
    
    double smin_line = 0; 
    double smax_line = 2;           // min/max line search steps size range
    int nmax_line = 10;                          // maximum number of line search steps
    double thr_line = 1e-4;                    // line search threshold
    
    // brent's minimization in one dimension
    int counters = 0;
    //fa
    Xa = Alpha + smin_line* dalpha;
    PSI(fa, Xa, Fv); 
    counters++;
    // fb
    Xb = dalpha * smax_line + Alpha;
    PSI(fb, Xb, Fv); 
    counters++;
    double seps = sqrt(2.220446049250313e-16);
    double c = 0.5*(3.0 - sqrt(5.0));// golden ratio
    double a = smin_line;
    double b = smax_line;
    double v =  a + c*(b-a);
    double w = v;
    double xf = v;
    double d = 0;
    double e = 0;
    double x = xf;
    Xc = Alpha + x*dalpha;
    PSI(fc, Xc, Fv);
    counters++;
    double fv = fc(0);
    double fw = fc(0);
    double xm = 0.5*(a + b);
    double tol1 = seps*abs(xf) + thr_line/3.0;
    double tol2 = 2.0*tol1;
    double si;
    double gs;
    double r;
    double q;
    double p;
    double sd;
    double xmin;
    while ( abs(xf-xm) > (tol2 - 0.5*(b-a)) )
    {
      gs = 1;
      if (abs(e) > tol1)
      {
	gs = 0;
	r =  (xf-w)*(fc(0)-fv);
	q = (xf-v)*(fc(0)-fw);
	p = (xf-v)*q-(xf-w)*r;
	q = 2.0*(q-r);
	if (q > 0.0)  
	  p = -p;
	q = abs(q);
	r = e;  e = d;
	
	if ( (abs(p)<abs(0.5*q*r)) && (p>q*(a-xf)) && (p<q*(b-xf)) )
	{
	  // Yes, parabolic interpolation step
	  d = p/q;
	  x = xf+d;
	  // f must not be evaluated too close to ax or bx
	  if (((x-a) < tol2) || ((b-x) < tol2))
	  {
	    si = sign(xm-xf) + ((xm-xf) == 0);
	    d = tol1*si;
	  }
	}
	else
	{
	  // Not acceptable, must do a golden section step
	  gs = 1;
	}
	
      }
      if (gs == 1)
      {
	//A golden-section step is required
	if (xf >= xm) 
	  e = a-xf;    
	else 
	  e = b-xf;
	d = c*e;
      }
      //The function must not be evaluated too close to xf
      si = sign(d) + (d == 0);
      sd = (abs(d)<tol1)?tol1:abs(d);
      x = xf + si * sd;
      Xc = dalpha*x + Alpha;
      PSI(fu, Xc, Fv);
      counters++;   
      // Update a, b, v, w, x, xm, tol1, tol2
      if (fu(0) <= fc(0))
      {
	if (x >= xf)
	  a = xf; 
	else 
	  b = xf;
	v = w;
	fv = fw;
	w = xf;
	fw = fc(0);
	xf = x;
	fc(0) = fu(0);
      }
      else
      {
	// fu > fc
	if (x < xf) 
	  a = x; 
	else 
	  b = x;
	if ( (fu(0) <= fw) || (w == xf) )
	{
	  v = w; fv = fw;
	  w = x; fw = fu(0);
	}
	else if ( (fu(0) <= fv) || (v == xf) || (v == w) )
	{ 
	  v = x; fv = fu(0);
	}
      }
      xm = 0.5*(a+b);
      tol1 = seps*abs(xf) + thr_line/3.0; 
      tol2 = 2.0*tol1;
      if (counters >= nmax_line)
	break;	
    }
    // check that endpoints are less than the minimum found
    if ( (fa(0) < fc(0)) && (fa(0) <= fb(0)) )
    {
      xf = smin_line; fc(0) = fa(0);
    }
    else if (fb(0) < fc(0))
    {
      xf = smax_line; fc(0) = fb(0);
    }
    fmin = fc(0);
    xmin = xf;
    Xc = dalpha*xmin + Alpha;
    Alpha = Xc;
    PSI(fu, Xc, Fv);
  }
  
  void GP_utils::updateAlpha() const
  {
    if (!AlphaUpStat())
    {
      updateKernel();
      irls();
      if (Chol_fail)
	return;;
      setAlphaUpStat(true);
    }
  }
  void GP_utils::mvmK_exact(mat alp) const
  {
    mvmK = K* alp;
  }
  void GP_utils::updatelikelihood(mat fval) const
  {
    switch (getLikelihoodType())
    {
      case likeL_Gaussian:
      {
	updateKernel();
// 	double sn2 = std::exp((double)2* hyperlf(0,0));
	double sn2 = hyperlf(0,0);
	mat ymmu(getNumData(),1);
	ymmu = yTarg - fval;
	lp = ymmu;
	lp =pow( ymmu, 2) * (-1/(2*sn2)) - std::log((double)2*M_PI*sn2)/2;
	dlp = (1/sn2) * ymmu;
	d2lp.ones();
	d2lp *= (1/sn2);
	d3lp.zeros();  
      }
      break;
      case likeL_WarpGauss:
      {
	updateKernel();
	warpingfunction();
	double sn2 = std::exp((double)2* hyperlf(getNumlikfpar()-1,0));
	mat ymmu(getNumData(),1);
	ymmu = gy - fval;
	lp = pow(ymmu, 2) *(-1/(2*sn2)) - log((double)2*M_PI*sn2)/2 + lgpy;
	dlp = (1/sn2) * ymmu;
	d2lp.ones();
	d2lp *= (1/sn2);
	d3lp.zeros();  
      }
      break;
    }
  }
  
  void GP_utils:: warpingfunction() const
  {
    switch(getWarpfunc())
    {
      case tanh1:
      {
	
	gy.resize (yTarg.n_rows, yTarg.n_cols);
	gy = yTarg;
	lgpy.resize (yTarg.n_rows, yTarg.n_cols);
	lgpy.zeros();
	mat gpy (yTarg.n_rows, yTarg.n_cols);
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, bi, ci;
	mat ti(gy.n_rows, gy.n_cols);
	mat dti(gy.n_rows, gy.n_cols);
	for (int i = 0; i <m; i++)
	{
	  ai = exp(hyperlf(i));
	  bi = exp(hyperlf(i+m));
	  ci = hyperlf(i + 2*m);
	  
	  ti = tanh((yTarg + ci)*bi);
	  
	  dti = pow(ti, 2)*-1 + 1;
	  gy += ai*ti;
	  dti *= (ai*bi);
	  gpy += dti;
	}
	lgpy = log(gpy);
      }
      break;
      case rbf:
      {
	
	mat ti(yTarg.n_rows, yTarg.n_cols);
	mat B(yTarg.n_rows, yTarg.n_cols);
	mat dti(yTarg.n_rows, yTarg.n_cols);
	mat gpy (yTarg.n_rows, yTarg.n_cols);
	gy.resize (yTarg.n_rows, yTarg.n_cols);
	gy = yTarg;
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, sigmai, ci, mi;
	for (int i = 0; i <m; i++)
	{
	  ai = exp(hyperlf(i));
	  sigmai = exp(hyperlf(i+m));
	  ci = exp(-hyperlf(i + 2*m));
	  mi = yTarg.max();
	  ci = max(mi, ci);
	  ti = yTarg - ci;
	  dti = ti;
	  ti = (pow(ai,2))*exp(pow(ti, 2) *(-1/pow(sigmai,2)));
	  B = ti;
	  dti = (dti*(-2/pow(sigmai,2)))% B;
	  gy += ti;
	  gpy += dti;
	}
	lgpy = log(gpy);
      }
      break;
      case srbf:
      {
	mat ti(yTarg.n_rows, yTarg.n_cols);
	mat ti2(yTarg.n_rows, yTarg.n_cols);
	mat B(yTarg.n_rows, yTarg.n_cols);
	mat dti(yTarg.n_rows, yTarg.n_cols);
	mat dti2(yTarg.n_rows, yTarg.n_cols);
	mat gpy (yTarg.n_rows, yTarg.n_cols);
	gy.resize (yTarg.n_rows, yTarg.n_cols);
	gy = yTarg;
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, sigmai, ci;
	for (int i = 0; i <m; i++)
	{
	  ai = hyperlf(i);
	  sigmai = hyperlf(i+m);
	  ci = hyperlf(i + 2*m);

	  ti = pow((yTarg - ci), 2);
	  dti = ti;
	  ti = (pow(ai,2))*exp((-1/pow(sigmai,2))* ti);
	  B = ti;
	  ti2 = erfc(-1*abs(yTarg - ci));
	  ti = ti % ti2;
	  dti2 = yTarg - ci;
	  dti = (exp(-1*dti)*(-2/sqrt(M_PI))) % B;
	  
	  umat ind = dti2 > 0.0;
	  dti.elem(ind) = -1 *dti.elem(ind);
	  

	  dti2 = (((dti2*(-2/pow(sigmai,2))) % B) % ti2);

	  dti += dti2;

	  gy += ti;

	  gpy += dti;
	  if(any(vectorise(gpy) < 0))
	    cout<<"Error" <<endl;
	  if(any(vectorise(ti2) < 0))
	    cout<<"Error" <<endl;
	  
	}
	lgpy = log (gpy);
      }
      break;
    }
  }
  // In estiamating new data
  void GP_utils:: warpingfunction(const mat &yy, mat &gyy, mat &lgpyy) const
  {
    switch(getWarpfunc())
    {
      case tanh1:
      {
	
	mat ti(gyy.n_rows, gyy.n_cols);
	mat dti(gyy.n_rows, gyy.n_cols);
	mat gpy (yy.n_rows, yy.n_cols);
	gyy = yy;
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, bi, ci;
	for (int i = 0; i <m; i++)
	{
	  ai = exp(hyperlf(i));
	  bi = exp(hyperlf(i+m));
	  ci = hyperlf(i + 2*m);
	  ti = tanh((yy+ci) * bi); 
	  dti = -1 * pow(ti, 2) + 1;
	  gyy += ai*ti;
	  gpy += dti*(ai*bi);
	}
	lgpyy = log(gpy);
      }
      break;
      case rbf:
      {
	
	mat ti(gyy.n_rows, gyy.n_cols);
	mat B(gyy.n_rows, gyy.n_cols);
	mat dti(gyy.n_rows, gyy.n_cols);
	mat gpy (yy.n_rows, yy.n_cols);
	gyy = yy;
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, sigmai, ci, mi;
	for (int i = 0; i <m; i++)
	{
	  ai = exp(hyperlf(i));
	  sigmai = exp(hyperlf(i+m));
	  ci = exp(-hyperlf(i + 2*m));
	  mi = yTarg.max();
	  ci = max(mi, ci);
	  ti = yy - ci;
	  dti = ti;
	  ti = (pow(ai,2))* exp((-1/pow(sigmai,2)) *pow(ti, 2));
	  B = ti;
	  dti = ((-2/pow(sigmai,2)) * dti) % ti;
	  gyy += ti;
	  gpy += dti;
	}
	lgpyy = log(gpy);
      }
      break;
      case srbf:
      {
	mat ti(gyy.n_rows, gyy.n_cols);
	mat ti2(gyy.n_rows, gyy.n_cols);
	mat B(gyy.n_rows, gyy.n_cols);
	mat dti(gyy.n_rows, gyy.n_cols);
	mat dti2(gyy.n_rows, gyy.n_cols);
	mat gpy (yy.n_rows, yy.n_cols);
	gyy = yy;
	gpy.ones();
	int m = getNumlikfpar()/3;
	double ai, sigmai, ci;
	for (int i = 0; i <m; i++)
	{
	  ai = hyperlf(i);
	  sigmai = hyperlf(i+m);
	  ci = hyperlf(i + 2*m);
	  ti = pow((yy - ci), 2);
	  dti = ti;
	  ti = (pow(ai,2)) * exp((-1/pow(sigmai,2))* ti);
	  B = ti;
	  ti2 = erfc(-1 * abs(yy - ci));
	  ti = ti % ti2;
	  dti2 = yy - ci;
	  dti = (exp(-1 * dti)* (-2/sqrt(M_PI))) % B;
	  
	  umat ind = dti2 > 0.0;
	  dti.elem(ind) = -1 *dti.elem(ind);
	  dti2 = ((dti2 *(2/pow(sigmai,2)))% B) % ti2;

	  dti += dti2;

	  gyy += ti;
	  if(any(vectorise(gpy)< 0))
	    cout<<"Error" <<endl;
	  if(any(vectorise(ti2)< 0))
	    cout<<"Error" <<endl;	  

	  gpy += dti;
	}

	lgpyy = log(gpy);
      }
      break;
    }
  }  
  
  void GP_utils::inverse_warpingfunction(const mat &Z, mat &G) const
  {
    switch (getWarpfunc())
    {
      case tanh1:
      case rbf:
      {
	mat y(Z.n_rows, Z.n_cols);
	mat ylow(Z.n_rows, Z.n_cols);
	mat yup(Z.n_rows, Z.n_cols);
	mat gpyy(Z.n_rows, Z.n_cols);
	mat lgpyy = zeros<mat>(Z.n_rows, Z.n_cols);
	mat gyy = zeros<mat>(Z.n_rows, Z.n_cols);
	mat y_tmp(Z.n_rows, Z.n_cols);
	mat gyz(Z.n_rows, Z.n_cols);
	double dd;

	y = Z;

	warpingfunction(y, gyy, lgpyy);

	gyy -= Z;

	mat Zabs = abs(Z);
	double dz = Zabs.max();
	while (any(vectorise(gyy) > 0))
	{

	  umat ind = gyy > 0.0;
	  y.elem(ind) = y.elem(ind) - dz;
	  
	  warpingfunction(y, gyy, lgpyy);

	  gyy -= Z;
	}

	ylow = y;
	y = Z;
	warpingfunction(y, gyy, lgpyy);

	gyy -= Z;
	Zabs = abs(Z);
	dz = Zabs.max();

	while (any(vectorise(gyy) < 0))
	{

	  umat ind = gyy < 0.0;
	  y.elem(ind) = y.elem(ind) + dz;
	  warpingfunction(y, gyy, lgpyy);

	  gyy -= Z;
	} 

	yup = y;
	for (int n = 0; n < 12; n++)
	{

	  mat Zabs = abs(gyy);
	  double dd = Zabs.max();
	  if (dd < sqrt(__DBL_EPSILON__))
	    break;

	  y = ylow;

	  y += yup;

	  y = 0.5*y;
	  warpingfunction(y, gyy, lgpyy);

	  gyy = gyy - Z;

	  umat ind = gy < 0;
	  ylow.elem(ind) = y.elem(ind);      

	  ind = gy > 0;
	  ylow.elem(ind) = y.elem(ind);
	  
	}
	for (int n = 0; n < 12; n++)
	{
	  warpingfunction(y, gyy, lgpyy);

	  gpyy = lgpyy;

	  gpyy = exp(gpyy);

	  y_tmp = gyy;

	  y_tmp -= Z;

	  y_tmp%(1/gpyy);

	  y -= y_tmp;

	  umat ind = y < ylow;
	  y.elem(ind) = ylow.elem(ind);

	  ind = y > yup;
	  y.elem(ind) = yup.elem(ind);

	  gyz = gyy;

	  gyz -= Z; 
	  gyz = abs(gyz);
	  dd = gyz.max();
	  if (dd < sqrt(__DBL_EPSILON__))
	    break;
	}
	if (dd>sqrt(__DBL_EPSILON__)) 
	  cout<<"Not converged : "<< dd<< endl;
	G = y;
      }
      break;
      case srbf:
      {
	mat lny(Z.n_rows, Z.n_cols);

	lny = Z;
	int m = getNumlikfpar()/3;
	double ai, sigmai, ci, mi;
	for (int i = 0; i <m; i++)
	{
	  ai = exp(hyperlf(i));
	  sigmai = exp(hyperlf(i+m));
	  ci = exp(-hyperlf(i + 2*m));
	  mi = yTarg.max();
	  ci = max(mi, ci);
	  lny = lny *(1/pow(ai,2));
	  lny = log(lny);

	  lny = -1 *pow(sigmai,2)* lny;
	  
	  lny = sqrt(lny);
	  
	  G = lny;
	  
	  G = G + ci;
	}
	
      }
      break;
    }
  }
  void GP_utils::updatelikelihood() const
  {
    if (!LikelihoodUpStat())
    {
      switch (getLikelihoodType())
      {
	case likeL_Gaussian:
	{
	  
	  updateKernel();
	  // 	  double sn2 = std::exp((double)2* hyperlf(0,0));
	  double sn2 = hyperlf(0,0);
	  mat ymmu(getNumData(),1);
	  ymmu = yTarg - yhat;

	  lp = (-1/(2*sn2)) *pow(ymmu, 2) - log((double)2*M_PI*sn2)/2;

	  dlp = (1/sn2) * ymmu;
	  d2lp.ones();
	  d2lp *= (1/sn2);
	  d3lp.zeros();  
	}
	break;
	case likeL_WarpGauss:
	{
	  updateKernel();
	  warpingfunction();
	  double sn2 = std::exp((double)2* hyperlf(getNumlikfpar()-1,0));
	  mat ymmu(getNumData(),1);

	  ymmu = gy - yhat;

	  lp = (-1/(2*sn2)) *pow(ymmu, 2) - log((double)2*M_PI*sn2)/2;

	  dlp = (1/sn2) *ymmu;
	  d2lp.ones();
	  d2lp = (1/sn2) * d2lp;
	  d3lp.zeros();  
	}
	break;
      }
      
      setLikelihoodUpStat(true);
    }
  }
  
  double GP_utils::solve_chol(const mat Lc, mat& Xr, const mat dB) const
  {
    Xr = solve(trimatl(Lc.t()), dB);
    Xr = solve(trimatu(Lc), Xr);
  }
  void GP_utils::updateGlikelihood() const
  {
    switch (getLikelihoodType())
    {
      case likeL_Gaussian:
      {
	
// 	double sn2 = std::exp((double)2* hyperlf(0,0));
	double sn2 = hyperlf(0,0);
	mat ymmu(getNumData(),1);
	ymmu = yTarg - yhat;

	lp_dhyp = (1/sn2) *pow(ymmu ,2) - 1;

	dlp_dhyp = (-2/sn2) * ymmu;
	d2lp_dhyp.ones();
	d2lp_dhyp = (2/sn2) * d2lp_dhyp;
      }
      break;
      case likeL_WarpGauss:
      {
	cout <<"Grad likeL_WarpGauss not yet implemented for GP_utils. \n";
	exit(1);
      }
    }
  }
  void GP_utils::ldB2_exact(mat WW, mat r, mat &QQ) const
  {
    mat Ide = zeros<mat>(getNumData(), getNumData());
    mat rSw = zeros<mat>(getNumData(), 1);
    Sw = sqrt(WW);
    Ide.eye();
    Lchol.zeros();
    Lchol = Sw * Sw.t();
    Lchol = Lchol % K + Ide;
    bool b = chol(Lchol, Lchol);
    if (b == false)
    {
      Chol_fail = true;
      return;
    }
    else
      Chol_fail = false;
    
    rSw = Sw % r;
    solve_chol(Lchol, QQ, rSw);
    QQ %= Sw;
  }
  void GP_utils::ldB2_exact() const
  {
    mat Ide = zeros<mat>(getNumData(), getNumData());

    Sw = sqrt(d2lp);
    Ide.eye();
    Lchol.zeros();
    Lchol = Sw * Sw.t();
    Lchol = Lchol % K + Ide;
    bool b = chol(Lchol, Lchol);
    if (b == false)
    {
      Chol_fail = true;
      return;
    }
    else
      Chol_fail = false;
    Lchol_db2 = 0;
    
    Lchol_db2 = accu(log(Lchol.diag()));
    
  }
  void GP_utils::ldB2_exact_drive(mat dWs)
  {
    if (any(vectorise(d2lp) < 0))
      printf("Warining negative value in derivative\n");
    else
    {
      
      
      mat dsw (getNumData(), getNumData());
      mat invsw (getNumData(),1);
      mat dw (getNumData(), getNumData());
      
      // make diagonal
      dsw = Sw.diag();
      solve_chol(Lchol, Q, dsw);
      invsw = 1/Sw;
      for (int ii = 0; ii < Q.n_rows; ii++)
      {
	Q.row(ii) = Q.row(ii) * invsw(ii);
      }
      dw = Q % K;
      dWs = sum(dw, 1)*0.5;
    }
    
    
  }
  
  void GP_utils::_ComputeK_NewData(mat &kX, const mat& Xin) const
  {
    
    mat CR(Xinp.n_rows, Xin.n_rows);
    KerenlW->computeK(Xinp, Xin, kX, CR);
    
  }
  // compute diagonal
  void GP_utils::_ComputeDiag_NewData(mat &kD, const mat& Xin) const
  {
    
    KerenlW->diag_Compute(kD, Xin);
    
  }
  
  void GP_utils::_postMean(mat& mu, const mat& kX) const
  {
    mat mfval = zeros<mat>(kX.n_cols, 1);
    updateAlpha();
    updateMean(mfval);
    
    for (unsigned int i = 0; i < kX.n_cols; i++)
    {
      for (unsigned int j = 0; j < getOutDim(); j++)
      {
	mu(i, j) = dot(Alpha.col(j), kX.col(i));
      }
    }
    mu += mfval;
  } 
  void GP_utils::_postVar(mat& varSigma, mat& kX, const mat& Xin) const
  {
    mat Wh (getNumData(), 1);
    mat LKs (getNumData(), 1);
    mat kD(Xin.n_rows,1);
    _ComputeDiag_NewData(kD, Xin);
    updateKernel();
    logLikelihood();
    
    
    
    
    Wh = sqrt(d2lp);
    LKs = kX;
    for (int ii = 0; ii < LKs.n_rows; ii++)
    {
      LKs.row(ii) = LKs.row(ii) * Wh(ii);
    }
    solve_chol(Lchol, LKs, LKs);
    //     LKs.bsxfun_times(Wh);
    for (int ii = 0; ii < LKs.n_rows; ii++)
    {
      LKs.row(ii) = LKs.row(ii) * Wh(ii);
    }
    LKs %= kX;
    varSigma = kD.t();
    varSigma -= sum(LKs, 0);
    varSigma = varSigma.t();
    double Zr = 0.0;
    uvec ind = varSigma < 0;
    varSigma.elem(ind) = zeros<mat>(ind.n_rows, ind.n_cols);
  }
  void GP_utils::posteriorMean(mat& mu, const mat& Xin) const
  {
    updateAlpha();
    int alRows = 0;
    alRows = getNumData();
    
    mat kX(alRows, Xin.n_rows);
    _ComputeK_NewData(kX, Xin);
    
    _postMean(mu, kX);
  }
  void GP_utils::posteriorMeanVar(mat& mu, mat& varSigma, const mat& Xin) const
  {
    
    int alRows = 0;
    alRows = getNumData();
    mat kX(alRows, Xin.n_rows);
    
    _ComputeK_NewData(kX, Xin);
    _postMean(mu, kX);
    
    _postVar(varSigma, kX, Xin); // destroys kX through in place operations.
    
    switch (getLikelihoodType())
    {
      case likeL_Gaussian:
      {
	// apply gaussian likelihood hyperparmeter (noise)
	for (unsigned int j = 0; j < getOutDim(); j++)
	{
// 	  double hyperVal = exp(hyperlf(getNumlikfpar()-1)*2);
	  double hyperVal = hyperlf(getNumlikfpar()-1);
	  if (hyperVal != 1.0)
	  {
	    varSigma += hyperVal;
	  }
	}
      }
      break;
      case likeL_WarpGauss:
      {
	// apply gaussian likelihood hyperparmeter (noise)
	int N = 20;
	mat Z(Xin.n_rows, N);
	mat Z2(Xin.n_rows, N);
	for (unsigned int j = 0; j < getOutDim(); j++)
	{
	  double hyperVal = exp(hyperlf(getNumlikfpar()-1)*2);
	  if (hyperVal != 1.0)
	  {
	    varSigma += hyperVal;
	  }
	}
	// Gauss-Hermite
	mat weight(1,N);
	vec abscissas(N,1); 
	mat Onesmu = ones<mat>(1, N);
	mat SvarSigma(Xin.n_rows, N);
	Gauher(weight,  abscissas, N);
	SvarSigma = varSigma;
	SvarSigma = sqrt(SvarSigma);
	Z = SvarSigma * abscissas.t();
	Z2 = mu * Onesmu;
	Z += Z2;
	mat G(Z.n_rows, Z.n_cols);
	Z.save("Z.txt", csv_ascii);
	inverse_warpingfunction(Z, G);
	
	mu = G*weight.t();
	Z2 = -1*Z2;
	Z2 += G;
	Z2 = pow(Z2, 2);
	varSigma = Z2*weight.t();
      }
      break;
    }
  }
  void GP_utils::Gauher(mat &weight, vec& abscissas, double N) const// Gaussian-Hermite Quadrature
  {
    vec b = regspace< vec>(0,  N-1);       // 0,  1, ...,   9
    mat Dg0(N,N);
    mat Dg1(N,N);
    
    b = sqrt(0.5*b);
    Dg0 = b.diag(1);
    Dg1 = b.diag(-1);
    Dg0 += Dg1;
    
    int lwork = 3*N -1;
    eig_sym( abscissas, Dg0, Dg0) ;
    weight.row(0) = pow(Dg0.row(0), 2);
  }
  
  
  
  void GP_utils::updateKernel() const
  {
    if (!isUpdateK())
    {
      _updateKernel();
      
      setKUpdateStat(true);
    }
  }
  
  void GP_utils::_updateKernel() const
  {
    KerenlW->computeK(Xinp, Xinp, K, D2);
  }
  
 
  
  double GP_utils:: mvmK_exact( mat X, mat Z, const mat K) const
  { 
    mat tmp(1, 1);
    for (int i = 0; i < X.n_rows; i++)
    {
      tmp = X;
      Z(i, 0) = dot(tmp.row(0), K.col(i));
    }
  }
  void GP_utils:: updateMean() const
  { 
    mf (0, 0) = 0;
  }
  void GP_utils:: updateMean(mat &mfval) const
  { 
    mfval.zeros();
  }
  void GP_utils:: updateGMean() const
  { 
    dmf(0, 0) = 0;
  }
  double GP_utils::logLikelihood() const
  {
    
    mat ydif(getNumData(),1);
    mat nL = zeros<mat>(1,1);
    updateKernel();
    updateAlpha();
    if (Chol_fail)
      return std::numeric_limits<double>::quiet_NaN();
    mvmK_exact(Alpha);
    yhat = mvmK;
    updateMean();
    yhat += mf;
    setLikelihoodUpStat(false);
    updatelikelihood();
    ydif = 0.5*(yhat - mf);
    ldB2_exact();
    if (Chol_fail)
    {
      return std::numeric_limits<double>::quiet_NaN(); 
    }
    L = Alpha.t() * ydif - accu(lp) + Lchol_db2;
    return L[0];
    setlogLikelihoodUpStat(true);
  }
  
  void GP_utils::dhyp(mat &dhat) const
  {
    mat R(getNumData(), getNumData());
    mat One = ones<mat>(1, Q.n_rows);
    QW = Q % (d2lp * One) - Alpha * Alpha.t() + dlp * dhat.t()* 2.0;
  }
  
  double GP_utils::GradLL(mat& g) const
  {
    if( !getlogLikelihoodUpStat() )
      logLikelihood();
    if (Chol_fail)
      return std::numeric_limits<double>::quiet_NaN();
    mat dfhat = zeros<mat>(getNumData(), 1);
    mat kmvm_dfhat = zeros<mat>(getNumData(), 1);
    mat rdfhat = zeros<mat>(getNumData(), 1);
    mat dahat = zeros<mat>(getNumData(), 1);
    mat alfadahat = zeros<mat>(getNumData(), 1);
    mat invLc = zeros<mat>(getNumData(), getNumData());
    mat XX0 = zeros<mat>(getNumData(), 1);
    mat olddW = zeros<mat>(getNumData(), 1);
    mat B =  zeros<mat>(getNumData(), 1);
    mat B0 = zeros<mat>(getNumData(), 1);
    mat g_tmp = zeros<mat>(1,1);
    mat g_tmp0 = zeros<mat>(1,1);
    
    mat dsw (getNumData(), getNumData());
    dsw.zeros();
    mat invsw (getNumData(),1);
    invsw.ones();
    mat dw (getNumData(), getNumData());
    
    
    if (any(vectorise(d2lp) < 0))
      printf("Warining negative value in derivative\n");
    else
    {
      // make diagonal
      dsw.diag() = Sw;
      solve_chol(Lchol, Q, dsw);
      invsw = (1/Sw) * ones(1, getNumData());
      Q = Q % ((1/Sw) * ones(1, getNumData()));
      dW = 0.5*sum(Q %K, 1);
    }
    
    
    dfhat = dW;
    olddW = dW;
    dfhat = dfhat%d3lp;
    kmvm_dfhat = K*dfhat;
    kmvm_dfhat %= Sw;
    
    dahat = kmvm_dfhat;
    dahat = solve(trimatl(Lchol.t()), kmvm_dfhat);
    dahat = solve(trimatu(Lchol), dahat);
    dahat = -1 * dahat % Sw + dfhat;
    dhyp(dahat);
    updateG();
    for (unsigned int i = 0; i < getNumlikfpar(); i++)
    {
      updateGlikelihood();
      dW = olddW;
      g_tmp0 = -1* dW.t() * d2lp_dhyp - accu(lp_dhyp);
      B = K * dlp_dhyp;
      B0 = B;
      B %= Sw;
      solve_chol(Lchol, B, B);
      B %= Sw;
      B = -1 * B + B0;
      g_tmp0 += -1.0*dfhat.t() * B;
      g_hyperlf(i) = g_tmp0[0];// getNumData();///
    }
    for (unsigned int i = 0; i < getNumMFpar(); i++)
    {
      alfadahat = dahat;
      alfadahat += Alpha;
      updateGMean();
      g_hypermf(i) = 0;
    }
    int counter = 0;
    
    
    for (unsigned int i = 0; i < KerenlW->getNPars(); i++)
    {
      g(0, counter) = g_param(0, i);
      counter++;
    }
    for (unsigned int i = 0; i < getNumlikfpar(); i++)
    {
      g(0, counter) = g_hyperlf(0, i);
      counter++;
    }
    for (unsigned int i = 0; i < getNumMFpar(); i++)
    {
      g(0, counter) = g_hypermf(0, i);
      counter++;
    }
    return L[0];
  }
  void GP_utils::updateG() const
  {
    
    unsigned int numKernParams = KerenlW->getNPars();
    unsigned int numParams = numKernParams;
    updateKernel();
    mat tempG(1, numKernParams);
    mat tempG2(1, numKernParams);
    mat Glike (1, getNumlikfpar());
    mat tmpV(getOutDim(), 1);
    g_param.zeros();
	for (unsigned int j = 0; j < getOutDim(); j++)
	{
	  
	  KerenlW->GetGrads(tempG, Xinp, Xinp, D2, QW);
	  
	  for (unsigned int i = 0; i < numKernParams; i++)
	  {
	    g_param(i) += tempG(i);
	  }
	}
  }
  
  
  
  void GP_utils::OptimisePars(unsigned int iters)
  {
    if (getVerbose() > 2)
    {
      cout << "Initial model:" << endl;
      ShowKernelPars(cout);
    }
    if (getVerbose() > 2 && getNumPars() < 40)
    setMaxIters(iters);
    Optimise();
    
    if (getVerbose() > 0)
      ShowKernelPars(cout);
  }
  
  void GP_utils::ShowKernelPars(ostream& os) const
  {
    cout << "Standard GP Model: " << endl;
    cout << "Optimiser: " << getDefaultOptimiserStr() << endl;
    cout << "Inference: " << getInferenceStr() << endl;
    cout << "likelihood function: " << getLiklihoodStr() << endl;
    if (likelihoodType == likeL_WarpGauss)
      cout << "Warping function: " << getWarpfuncstr() << endl;
    cout << "Mean function: " << getMeanTypeStr() << endl;
    cout << "Data Set Size: " << getNumData() << endl;
    cout << "Kernel Type: " << endl;
    KerenlW->ShowKernelPars(os);
    for (int i = 0; i < getNumlikfpar(); i++)
      cout << "likelihood hyperparmeters : " << (hyperlf(i)) << endl;
    for (int i = 0; i < getNumMFpar(); i++)
      cout << "likelihood hyperparmeters : " << exp(hypermf(i)) << endl;
    
    if (getVerbose())
      cout << "Log likelihood: " << logLikelihood() << endl;
  }
  
  void GP_utils::FromFile_GP_Params(istream& in)
  {
      setInf(ReadStrStrm(in,  "Inference"));
      setLikelihoodType(ReadIntStrm(in,  "likelihood"));    
      setMean(ReadStrStrm(in,  "MeanFunction"));    
      setNumData(ReadIntStrm(in, "numData"));
      setOutDim(ReadIntStrm(in, "outputDim"));
      setInpDim(ReadIntStrm(in, "inputDim"));
      setNumCovpar(ReadIntStrm(in, "NumHyperKernel"));
      setNumlikfpar(ReadIntStrm(in, "NumHyperLik"));
      setNumMFpar(ReadIntStrm(in, "NumHyperMean"));
      initialize_vars();
      KerenlW = ReadKerFromFile(in);
      g_param.resize(1, KerenlW->getNPars());
      double hypers;
      if (getNumlikfpar() > 0)
      {
	hyperlf.resize(getNumlikfpar(), 1);
	for (unsigned int i = 0; i < getNumlikfpar(); i++)
	{
	  hypers = (ReadDoubleStrm(in, "Hyperparams_likelihood"));
	  setHyperlfVal(hypers, i);
	}
      }
      if (getNumMFpar() > 0)
      {
	hypermf.resize(getNumMFpar(), 1);
	for (unsigned int i = 0; i < getNumMFpar(); i++)
	{
	  hypers = ReadIntStrm(in, "Hyperparams_meanfunction");
	  setHypermfVal(log(hypers), i);
	}
      }    
  }
  
  
  void GP_utils::ToFile_GP_Params(ostream& out) const
  {
    out << "Inference" << "=" << getInf() << endl;
    out << "likelihood" << "=" << getLikelihoodType() << endl;
    out << "MeanFunction" << "=" << getMean() << endl;
    out << "numData" << "=" << getNumData() << endl; 
    out << "outputDim" << "=" << getOutDim() << endl;
    out << "inputDim" << "=" << getInpDim() << endl;
    out << "NumHyperKernel" << "=" << KerenlW->getNPars() << endl;
    out << "NumHyperLik" << "=" << getNumlikfpar() << endl;
    out << "NumHyperMean" << "=" << getNumMFpar() << endl;
    KerenlW->StrmOut(out);
    double hypers;
    if (getNumlikfpar() > 0)
    {
      for (unsigned int i = 0; i < getNumlikfpar(); i++)
      {
// 	hypers = exp(getHyperlfVal(i));
	hypers = getHyperlfVal(i);
	out << "Hyperparams_likelihood" << "=" << hypers << endl;
      }
    }
    if (getNumMFpar() > 0)
    {
      for (unsigned int i = 0; i < getNumMFpar(); i++)
      {
	hypers = exp(getHypermfVal(i));
	out << "Hyperparams_meanfunction" << "=" << hypers << endl;
      }
    }
  }

  void writeGpToStream(const GP_utils& model, ostream& out)
  {
    model.StrmOut(out);
  }
  
  void writeGPFile(const GP_utils& model, const string modelFileName, const string comment)
  {
    model.WFile(modelFileName, comment);
  }
  
  GP_utils* readGpFromStream(istream& in)
  {
    GP_utils* ModelPred = new GP_utils();
    ModelPred->StrmIn(in);
    return ModelPred;
  }
  
  GP_utils* readGpFromFile(const string modelFileName, int verbosity)
  {
    if (verbosity > 0)
      cout << "Loading model file." << endl;
    ifstream in(modelFileName.c_str());
    if (!in.is_open()) 
    {cout << "Error in reading file name. \n";
      exit(1);}
      GP_utils* ModelPred;
      ModelPred = readGpFromStream(in);
      
      if (verbosity > 0)
	cout << "Model Info has been read.\n"; 
      in.close();
      ModelPred->setVerbose(verbosity);
      return ModelPred;
  }