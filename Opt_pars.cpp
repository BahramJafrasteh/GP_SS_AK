// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
#include "Opt_pars.h"

using namespace arma;
const double epsilon = numeric_limits<double>::epsilon();

void Opt_Algs::cauchy_point(const  mat g, const mat X,
			    const mat Wk, const mat Mk, mat &C, mat &xcp, mat & index_r, const double theta,
			    const double mnc
)
{
  
  double epsi = 1e-100;
  mat c(2*mnc, 1);
  c.zeros();
  index_r.resize(1, 1);
  int Dimension = getNumPars();
  //
  xcp.resize(1, Dimension);
  xcp = X;
  // t
  mat t(Dimension, 1);
  // d
  mat d(Dimension, 1);
  mat p(Dimension, 1);
  d = -g.t();
  for (int j = 0; j < Dimension; j++)
  {
    if ( g[j] < 0)
      t[j] = (X[j] - ub[j])/ g[j];
    else if (g[j] > 0)
      t[j] = (X[j] - lb[j])/ g[j];
    else
      t[j] = numeric_limits< double >::max(); 
    if (t[j] > - epsi && t[j] < epsi )
      d[j] = 0.0;
  }
  uvec indices = find(t > 0.0);
  mat F = t.elem(indices);
  p = Wk.t()*d;
  double fprime = accu(- d.t() * d);
  double fsec = accu(-theta*fprime - p.t()*Mk*p);
  double dt_min = -fprime / fsec;
  double t_old = 0.0;
  // min F and its index
  double mt = F.min();
  double b =  F.index_min();
  //   F[b] = numeric_limits::max();
  F.shed_row(b);
  index_r[0] = b;
  double dt = mt - t_old;
  int rowN = 1;
  while( dt_min >= dt && F.n_elem > 0 )
  {
    if ( d[b] > 0 )
      xcp[b] = ub[b];
    else if ( d[b] < 0 )
      xcp[b] = lb[b];
    double zb = xcp[b] - X[b];
    c += dt * p;
    fprime += dt * fsec + pow(g[b], 2) 
    + theta * g[b] * zb - g[b] * accu(Wk.row(b) * (Mk * c) );
    fsec += - theta * pow(g[b], 2) - 2.0 * g[b] * accu(Wk.row(b) * Mk * p)
    - pow(g[b], 2) * accu(Wk.row(b) * Mk * Wk.row(b).t());
    p += g[b] * Wk.row(b).t();
    d[b] = 0.0;
    dt_min = - fprime / fsec;
    t_old = mt;
    mt = F.min();
    b =  F.index_min();
    F.shed_row(b);
    rowN++;
    mat bb(1,1);
    bb[0] = b;
    index_r.insert_rows(rowN-1, bb);
    
    //     F[b] = numeric_limits::max();
    dt = mt - t_old;
  }
  
  dt_min = max (dt_min , 0.0);
  t_old += dt_min;
  for (int i = 0; i < Dimension; i++)
  {
    if ( t[i] >= mt)
      xcp[i] = X[i] + t_old*d[i];
  }
  for (int i = 0; i < F.n_rows; i++)
  {
    if ( t[i] == mt )
    {
      F.shed_row(i);
      rowN++;
      mat bb(1,1);
      bb[0] = i;
      index_r.insert_rows(rowN-1, bb);
    }
    //       F[i] = numeric_limits::max();
  }
  C = c + dt_min * p;
}

// a primal conjugate gradient method 
void Opt_Algs::Primal_Conjugate_grad(const mat index_r,
				     const mat xcp, const mat X, const mat Wk, const mat Mk,
				     const mat C, const mat g, const double theta, mat &direction
)
{
  int maxit = 50;
  direction.zeros();
  int Dimension = getNumPars();
  int Num_free_vars = Dimension - index_r.n_elem;
  if (Num_free_vars == 0)
  {
    direction = xcp - X;
    return;
  }
  mat Zk(Dimension, Dimension);
  Zk.zeros();
  Zk.diag().fill(1.0);
  for (int i = 0; i < Dimension; i++)
  {
    for (int j = 0; j < index_r.n_elem; j++)
      if (index_r[j] == i)
      {
	Zk(i, i) = 0.0;
	break;
      }
  }
  
  mat rc = Zk.t() * ( (g + theta * (xcp - X)).t()  - Wk*Mk*C );
  mat r = rc;
  mat p = -r;
  double Rho2 = accu(r.t() * r);
  double Rho1 = 0.0;
  int it = 0;
  while (norm(r) >= min(0.1 , sqrt(norm(rc))) * norm(rc) )
  {
    if (it > maxit)
      break;
    it++;
    mat pt = p;
    double alpha1 = -numeric_limits<double>::infinity();
    
    for (int i = 0; i < Dimension; i++)
    {
      if (pt[i] < 0)
	alpha1 = max ( alpha1, (lb[i] - xcp[i] - direction[i]) / pt[i]);
      else if ( pt[i] > 0 )
	alpha1 = max ( alpha1, (ub[i] - xcp[i] - direction[i]) / pt[i]);
    }
    mat Bk = theta * eye<mat>(Dimension,Dimension) - Wk * Mk * Wk.t();
    mat q = Bk * p;
    double alpha2 = Rho2 / accu(p.t() * q);
    if (alpha2 > alpha1)
    {
      direction += alpha1*p.t();
      break;
    }
    else
    {
      direction += alpha2 * p.t();
      r += alpha2 * q;
      Rho1 = Rho2;
      Rho2 = accu(r.t() * r);
      double beta = Rho2 / Rho1;
      p = -r + beta * p;
    }
  }
}
//R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm 
// for Bound Constrained Optimization, (1995), 
// SIAM Journal on Scientific and Statistical Computing , 
// 16, 5, pp. 1190-1208.
void Opt_Algs::LBFGSOptimise()
{
  
  mat index_r(1 , 1);
  int Dimension = getNumPars();
  lb.resize(1, Dimension);
  ub.resize(1, Dimension);
  lb.ones();
  lb = lb * 1e-4;
  ub = 6 * ub.ones();
  // maximum number of correction pairs
  int nc = 1;
  int mnc = 6;
  // Wk
  mat Wk(Dimension, 2*nc);
  // Mk
  mat Mk(2*nc, 2*nc);
  mat Dk (nc, nc);
  mat Lk(nc, nc);
  // Yk
  mat Yk(Dimension, nc);
  // Sk
  mat Sk(Dimension, nc);
  double theta = 0.9;
  
  mat C(2*mnc, 1);
  C.zeros();
  // initial X
  mat X0(1, Dimension);
  mat g(1, Dimension);
  get_GP_Pars(X0);
  set_GP_Pars(X0);
  mat X = X0;
  //
  mat search_direction (1, Dimension);
  // Gradient and objective val
  double fx = Grad_Values(g);
  mat xcp = X0;
  Mk.zeros();
  Dk.zeros();
  Dk(nc-1, nc-1) = accu(X0 * g.t());
  Yk.col(nc-1) = g.t();
  Sk.col(nc-1) = X0.t();
  Wk.submat(0, nc-1, Dimension-1, nc-1) = g.t();
  Wk.submat(0, 2*nc-1, Dimension-1, 2*nc-1) = theta *X0.t();
  Lk = trimatl ( Sk.t() * Yk );
  
  Mk.submat(0 ,  0, nc-1, nc-1) = -Dk;
  Mk.submat(0 , nc, nc-1, 2*nc-1) = Lk.t();
  Mk.submat( nc, 0, 2*nc-1, nc-1) = Lk;
  Mk.submat(nc , nc, 2*nc-1, 2*nc-1) = theta* Sk.t() * Sk;
  Mk = inv(Mk);
  int Maxit = getMaxIters();
  int iter = 0; 
  double fnew = fx;
  mat gnew = g;
  mat Xnew = X;
  
  double final_steplength = 1;
  mat g_old = gnew;
  while (true)
  {
    iter++;
    mat gold = gnew;
    mat Xold = Xnew;
    cauchy_point(gold, X0,
		 Wk, Mk, C, xcp, index_r, theta,
		 nc);
    Primal_Conjugate_grad(index_r,xcp, X0, Wk, Mk,C, gold, theta, 
			  search_direction);
    //     LineSearch(fx, X0, gold, final_steplength, search_direction);
    Efficient_line_search( fx, X0, gold, search_direction,
			   final_steplength);
    Xnew = X0 + final_steplength* search_direction;
    bool violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      final_steplength /= 1.2;
      Xnew = X0 + final_steplength*search_direction;
      violate = ChkBndStat(Xnew, lb, ub);
      if ( final_steplength < epsilon )
      {
	Xnew = X0;
	final_steplength = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    double fnew = Grad_Values(gnew);
    if (fnew < fx)
    {
      X0 = Xnew;
      fx = fnew;
      g_old = gnew;
    }
    // yk = gk+1 - gk
    mat yk =  gnew - gold;
    // sk = xk+1 - sk
    mat sk = Xnew - Xold;
    
    if ( accu(sk.t() * yk ) <= epsilon * accu(yk.t()* yk) )
    {
      if (getVerbose() > 0)
	cout << "Iteration: " << iter << " -logL: " << fx << endl;
    if (iter >= Maxit)
      break;
      continue;
    }
    if (nc < mnc)
    {
      nc++;
      vec Diagonal = Dk.diag();
      mat newDiag = (sk * yk.t());
      Diagonal.insert_rows(Diagonal.n_rows, newDiag );
      Dk.resize(nc, nc);
      Dk.zeros();
      for (int i = 0; i < nc; i++)
	Dk(i, i) = Diagonal[i];
      Yk.insert_cols( nc-1, yk.t() );
      Sk.insert_cols( nc-1, sk.t() );
      Wk.resize(Dimension, 2*nc);
      Wk.submat(0, 0, Dimension-1, nc-1) = Yk;
      Wk.submat(0, nc, Dimension-1, 2*nc-1) = theta *Sk;
      Lk = trimatl ( Sk.t() * Yk );
      Mk.resize(2*nc, 2*nc);
      Mk.submat(0 ,  0, nc-1, nc-1) = -Dk;
      Mk.submat(0 , nc, nc-1, 2*nc-1) = Lk.t();
      Mk.submat( nc, 0, 2*nc-1, nc-1) = Lk;
      Mk.submat(nc , nc, 2*nc-1, 2*nc-1) = theta* Sk.t() * Sk;
      Mk = inv(Mk);
    }
    else{
      Dk(0, 0) = accu(sk * yk.t());
      Yk.col(0) = yk.t();
      Sk.col(0) = sk.t();
      Wk.submat(0, 0, Dimension-1, 0) = g.t();
      Wk.submat(0, mnc, Dimension-1, mnc) = theta *X0.t();
      Lk = trimatl ( Sk.t() * Yk );
      Mk.submat(0 ,  0, mnc-1, mnc-1) = -Dk;
      Mk.submat(0 , mnc, mnc-1, 2*mnc-1) = Lk.t();
      Mk.submat( mnc, 0, 2*mnc-1, mnc-1) = Lk;
      Mk.submat(mnc , mnc, 2*mnc-1, 2*mnc-1) = theta* Sk.t() * Sk;
      Mk = inv(Mk);      
      
    }
    theta = accu (yk * yk.t() ) / accu ( yk * sk.t() );
    if (iter >= Maxit)
      break;
    if (getVerbose() > 0)
      cout << "Iteration: " << iter << " -logL: " << fx << endl;
    
  }  
  set_GP_Pars(X0);
}
// from Numerical Optimization by Jorge Nocedal  Stephen J. Wright
void Opt_Algs::LineSearch(const double fold, const mat X, const mat g, double &final_steplength,
			  const mat direction)
{
  // line search parameters
  double alpha = 1e-4; // 0 < alpha < beta < 1
  double beta = 0.9;
  mat aD = abs(direction);
  double steplength_curr = 1e-4;
  double steplength_p = steplength_curr;
  uvec ind = find(aD < epsilon );
  for (int i = 0; i < ind.n_elem; i++)
  {
    aD[ind[i]] = numeric_limits<double>::infinity();
  }
  double steplength_max = 1/aD.max();
  steplength_max = max (steplength_max, 100.0);
  double inc = (steplength_max - steplength_curr)/10.0;
  
  mat gstep = g;
  double fstep = fold;//numeric_limits<double>::infinity();
  double fstep_old = fstep;
  
  double iterN = 0;
  
  while (true) 
  {
    steplength_p = steplength_curr;
    // Xk+1 = Xk + landa * dk
    mat Xstep = X +steplength_curr *direction;  
    ChkBnd(Xstep, lb, ub);
    set_GP_Pars(Xstep);
    double fstep = Grad_Values(gstep);
    
    if ( abs( accu(gstep.t() * direction) ) <= beta *abs( accu(g.t() * direction) ) )
    {
      final_steplength = steplength_curr;
      break;
    }
    steplength_curr = steplength_curr *inc;
    if (steplength_curr > steplength_max)
    {
      steplength_curr = steplength_max;
      break;
    }
    if ( ( fstep > fold + alpha* accu (g.t() * direction) || 
      fstep > fstep_old) && (iterN > 1) )
    {
      Zoom(steplength_p, steplength_curr, final_steplength,
	   fold, alpha, beta, X, g, direction);
      break;
    }
    
    if ( accu(gstep.t() * direction)  > 0.0 )
    {
      Zoom(steplength_curr, steplength_p, final_steplength,
	   fold, alpha, beta, X, g, direction);
      break;
    }    
    
    iterN++;    
    
  }
  
}

void Opt_Algs::Zoom( double &steplength_low, double &steplength_high,
		     double &final_steplength, const double f0, const double alpha,
		     const double beta, const mat X0, const mat g0, const mat& direction)
{
  double fstep_p = numeric_limits<double>::infinity();
  while (true)
  {
    // main zoom
    mat gstep = g0;
    double current_steplength = 0.1 * ( steplength_low + steplength_high );
    
    mat XStlow = X0 + steplength_low * direction;
    ChkBnd(XStlow, lb, ub);
    set_GP_Pars(XStlow);
    double fstep_low = ObjVal();
    
    
    mat Xnew = X0 + current_steplength * direction;
    ChkBnd(Xnew, lb, ub);
    set_GP_Pars(Xnew);
    double fstep = Grad_Values(gstep);
    
    
    
    if ( fstep > f0 + alpha* accu (g0.t() * direction) || fstep >= fstep_low)
      steplength_high = current_steplength;
    else
    {
      if ( abs( accu (gstep.t() * direction) ) <= -beta* accu (g0.t() * direction)  )
      {
	final_steplength = current_steplength;
	break;
      }
      
      if ( accu (gstep.t() * direction) * (steplength_high - steplength_low) >= 0.0  )
	steplength_high = steplength_low;
      
      steplength_low = current_steplength;
    }
    if ( abs(fstep_p - fstep) < 1e-3 )
    {   final_steplength = current_steplength;
      break;
    }
    fstep_p = fstep;
  }
}



//Numerical Optimization by Jorge Nocedal Stephen 
// J. Wright

void Opt_Algs::BFGSOptimize()
{
  
  int Dimension = getNumPars();
  lb.resize(1, Dimension);
  ub.resize(1, Dimension);
  lb.ones();
  lb = lb * 1e-4;
  ub = 6 * ub.ones();
  mat Hes(Dimension, Dimension);
  Hes.eye(Dimension, Dimension);
  
  
  // initial X
  mat X0(1, Dimension);
  mat g(1, Dimension);
  get_GP_Pars(X0);
  set_GP_Pars(X0);
  mat X = X0;
  
  //
  mat search_direction (1, Dimension);
  // Gradient and objective val
  double fx = Grad_Values(g);
  int Maxit = getMaxIters();
  iter = 0; 
  double fnew = fx;
  mat gnew = g;
  mat Xnew = X;
  Hes = eye<mat> (Dimension, Dimension)
  / accu(g * X.t() );
  double final_steplength = 1;
  mat g_old = gnew;
  while (true)
  {
    iter++;
    mat gold = gnew;
    mat Xold = Xnew;
    
    mat search_direction = - g *Hes ;
    //     LineSearch(fx, X0, gold, final_steplength, search_direction);
    Efficient_line_search( fx, X0, gold, search_direction,
			   final_steplength);
    Xnew = X0 + final_steplength* search_direction;
    bool violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      final_steplength /= 1.2;
      Xnew = X0 + final_steplength*search_direction;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( final_steplength < epsilon )
      {
	Xnew = X0;
	final_steplength = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    double fnew = Grad_Values(gnew);
    if (fnew < fx)
    {
      X0 = Xnew;
      fx = fnew;
      g_old = gnew;
    }
    // yk = gk+1 - gk
    mat yk =  gnew - gold;
    // sk = xk+1 - sk
    mat sk = Xnew - Xold;
    
    if ( iter ==1)
    {
      Hes = eye<mat> (Dimension, Dimension) * 
      accu(sk * yk.t() ) / accu(yk * yk.t() );
    }
    else {
      double rho = 1.0 / accu(yk * sk.t() );
      Hes = (eye<mat>(Dimension, Dimension) -
      rho * sk.t() * yk ) * Hes * (eye<mat>(Dimension, Dimension) -
      rho * yk.t() * sk ) + rho * sk.t() * sk;
    }
    if (iter >= Maxit)
      break;
    if (getVerbose() > 0)
      cout << "Iteration: " << iter << " -logL: " << fx << endl;    
  }  
  set_GP_Pars(X0);
}



// Efficient line search algorithm by F. A.POTRA and Y.SHI
void Opt_Algs::Efficient_line_search(
  const double fxk,
  const arma::mat X,
  const mat gk,
  mat &sk,
  double &final_steplength
)
{
  // six given user parameters
  double rho = 1e-14;  // 0<rho<0.5
  double sig = 0.99;  //   p<sig<1
  double J = 2.0 ;  // 2.0 < J<=  9.0
  double tau1 = 1e-14;   //  0.0 <tau1 < tau2< 0.5
  double tau2 = 0.49;   //  0.0 <tau1 < tau2< 0.5
  double tau3 = 2.1; //	tau3 > 2
  // max iter line search
  int maxls = 4;
  
  // steplength alpha
  
  mat XX = abs(sk);
  double maxX = accu(XX) / XX.n_elem;
  double steplength = 1.0;
  double a = 0.0;
  double b = steplength;
  bool returnflg = false;
  const double f0 = fxk;
  double global_val = f0;
  // sk : search direction
  double fprim0 = accu ( gk.t() * sk);
  mat gnew = gk;
  /*if (fprim0 > 0)
   *  {final_steplength = steplength;
   *    return;}*/
  if (fail_pre_bfgs)
    steplength = -1.0;
  mat Xnew = X + steplength*sk;
  bool violate = ChkBndStat(Xnew, lb, ub);
  while (violate)
  {
    steplength /= 1.2;
    Xnew = X + steplength*sk;
    violate = ChkBndStat(Xnew, lb, ub);
          if ( steplength < epsilon )
      {
	steplength = 0.0;
	Xnew = X;
	break;
      }
  }
  set_GP_Pars(Xnew);
  const double f1 = Grad_Values(gnew);
  if (f1 < global_val)
  {
    final_steplength = 1.0;
    global_val = f1;
  }
  double fa;
  double fb;
  if ( f1 > f0 + rho * fprim0)
  {
    a = 0.0;
    b = steplength;
    Xnew = X + a*sk;
    set_GP_Pars(Xnew);
    fa = ObjVal();
    if (fa < global_val)
    {
      final_steplength = a;
      global_val = fa;
    }
    Xnew = X + b*sk;
      bool violate = ChkBndStat(Xnew, lb, ub);
  while (violate)
  {
    b /= 1.2;
    Xnew = X + b*sk;
    violate = ChkBndStat(Xnew, lb, ub);
          if (b < epsilon )
      {
	Xnew = X;
	b = 0.0;
	break;
      }
  }
    set_GP_Pars(Xnew);
    fb = ObjVal();
    if (fb < global_val)
    {
      final_steplength = b;
      global_val = fb;
    }
  }
  else{
    if (sig > 0.5)
    {
      if ( f1 >= f0 + sig * fprim0 ) 
      {
	final_steplength = 1.0;
	returnflg = true;
      }
      
    }
    else{
      double fprim1 = accu(gnew.t() * sk);
      if ( fprim1 >= sig * fprim0 )
      {
	final_steplength = 1.0;
	returnflg = true;
      }
    }
    
    if (returnflg)
    {
      if (global_val <= f0)
	fail_pre_bfgs = false;
      else
	fail_pre_bfgs = true;
      return;
    }
    
    // Step 2
    double an = 1.0;
    double bn = J;
    
    
    Xnew = X + an*sk;
    violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      an /= 1.2;
      Xnew = X + an*sk;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( an < epsilon )
      {
	Xnew = X;
	an = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    fa = ObjVal();
    if (fa < global_val)
    {
      final_steplength = an;
      global_val = fa;
    }
    Xnew = X + bn*sk;
    violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      bn /= 1.2;
      Xnew = X + bn*sk;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( bn < epsilon )
      {
	Xnew = X;
	bn = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    fb = ObjVal();
    if (fb < global_val)
    {
      final_steplength = bn;
      global_val = fb;
    } 
    
    while (true)
    {
      
      // step 2a
      if ( fb > fa + (bn - an) * rho * fprim0 )
      {
	a = an;
	b = bn;
	break;
      }
      // step 2b
      else if ( fb >= fa + (bn - an) * sig * fprim0 )
      {
	final_steplength = bn;
	returnflg = true;
	break;
      }
      // step 2c
      else{
	an = bn;
	bn = J * bn;
	Xnew = X + an*sk;
	violate = ChkBndStat(Xnew, lb, ub);
	while (violate)
	{
	  an /= 1.2;
	  Xnew = X + an*sk;
	  violate = ChkBndStat(Xnew, lb, ub);
	  an /= 2.0;
	        if ( an < epsilon )
      {
	Xnew = X;
	an = 0.0;
	break;
      }
	}
	if (fa != fa || fb != fb)
	{
	  returnflg = true;
	  break;
	}
	set_GP_Pars(Xnew);
	fa = ObjVal();
	if (fa < global_val)
	{
	  final_steplength = an;
	  global_val = fa;
	}
	Xnew = X + bn*sk;
	violate = ChkBndStat(Xnew, lb, ub);
	while (violate)
	{
	  bn /= 1.2;
	  Xnew = X + bn*sk;
	  violate = ChkBndStat(Xnew, lb, ub);
	        if ( bn < epsilon )
      {
	Xnew = X;
	bn = 0.0;
	break;
      }
	}
	set_GP_Pars(Xnew);
	fb = ObjVal();
	if (fb < global_val)
	{
	  final_steplength = bn;
	  global_val = fb;
	}
      }  
      
      
    }
  }
  if (returnflg)
  {
    if (global_val <= f0)
      fail_pre_bfgs = false;
    else
      fail_pre_bfgs = true;
    return;
  }
  
  // step 3
  double an = a;
  double bn = b;
  double cn = an;
  double deltan = 0.0;
  double it = 0;
  double fcn_old;
  while (true && it < maxls)
  {
    it++;
    // step 3a
    // two-point interpolation formula
    double lowv = an + tau1 * (bn - an);
    double highv = an + tau2 * (bn - an);
    
    
    Xnew = X + lowv*sk;
    bool violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      tau1 /= 1.2;
      lowv = an + tau1 * (bn - an);
      Xnew = X + lowv*sk;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( lowv < epsilon )
      {
	Xnew = X;
	lowv = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    mat glow = gk;
    mat ghigh = gk;
    double flow = Grad_Values(glow);
    if (flow < global_val)
    {
      final_steplength = lowv;
      global_val = flow;
    }
    Xnew = X + highv*sk;
    violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      tau2 /= 1.1;
      highv = an + tau2 * (bn - an);
      Xnew = X + highv*sk;
      violate = ChkBndStat(Xnew, lb, ub);
      if (tau2 >= tau1)
	break;
            if ( highv < epsilon )
      {
	Xnew = X;
	highv = 0.0;
	break;
      }
    }
    
    set_GP_Pars(Xnew);
    
    double fhigh = Grad_Values(ghigh);
    if (fhigh < global_val)
    {
      final_steplength = highv;
      global_val = fhigh;
    }
    double fprimlow = accu(glow.t() *sk);
    double fprimhigh = accu(ghigh.t() *sk);
    double x0 = 0.25 * (lowv + highv);
    double y0 = (flow + (x0 - lowv)* fprimlow) * (highv - x0) / (highv - lowv) +
    (fhigh + (x0 - highv)* fprimhigh) * (x0 - lowv) / (highv - lowv);
    double x1 = 0.5 * (lowv + highv);
    double y1 = (flow + (x1 - lowv)* fprimlow) * (highv - x1) / (highv - lowv) +
    (fhigh + (x1 - highv)* fprimhigh) * (x1 - lowv) / (highv - lowv);  
    double x2 = 0.75 * (lowv + highv);
    double y2 = (flow + (x2 - lowv)* fprimlow) * (highv - x2) / (highv - lowv) +
    (fhigh + (x2 - highv)* fprimhigh) * (x2 - lowv) / (highv - lowv);
    double minf = min (y0, y1);
    minf = min (minf , y2);
    if (minf == y0)
      cn = x0;
    else if (minf == y1)
      cn = x1;
    else if (minf == y2)
      cn = x2;
    
    Xnew = X + cn*sk;
    violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      cn /= 1.1;
      Xnew = X + cn*sk;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( cn < epsilon )
      {
	Xnew = X;
	cn = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    double fcn = ObjVal();
    if (fcn < global_val)
    {
      final_steplength = cn;
      global_val = fcn;
    }
    if ( it == 1)
      deltan = abs ( ( 
      (fb - fcn) / (bn - cn) - (fcn - fa)/(cn - an) ) / (bn - an) );
    
    // step 3b
    if ( fcn <= fa + (cn - an) * rho * fprim0 &&
      fcn >= fa + (cn - an) * sig * fprim0)
    {
      final_steplength = cn;
      returnflg = true;
      break;
    }
    // step 3c
    else{
      deltan = abs ( ( 
      (fb - fcn) / (bn - cn) - (fcn - fa)/(cn - an) ) / (bn - an) );
    }
    // step 3d
    if (fcn <= fa + (cn - an) *rho * fprim0 )
    {
      if ( (rho - sig) * fprim0 >= tau3 * (bn - an) * deltan )
      {
	steplength = cn;
      }
      else{
	an = cn;
	Xnew = X + an*sk;
	ChkBnd(Xnew, lb, ub);
	set_GP_Pars(Xnew);
	fa = ObjVal();
	if (fa < global_val)
	{
	  final_steplength = an;
	  global_val = fa;
	}
      }
    }
    else {
      // step 3e
      if ((rho - sig) * fprim0 >= tau3 * (bn - an) * deltan &&
	an > 0
      )
      {
	final_steplength = an;
	returnflg = true;
	break;
      }
      else{
	bn = cn;
	Xnew = X + bn*sk;
	ChkBnd(Xnew, lb, ub);
	set_GP_Pars(Xnew);
	fb = ObjVal();
	if (fcn < global_val)
	{
	  final_steplength = bn;
	  global_val = fb;
	}
      }
    }
  }  
  
  if (global_val < f0)
    fail_pre_bfgs = false;
  else
    fail_pre_bfgs = true;
  if (!returnflg)
  {
    final_steplength = steplength;
  }
  
  
}


// A scaled conjugate gradient for fast supervised learning

void Opt_Algs::scgOptimise()
{
  if(getVerbose()>2)
  {
    cout << "Scaled Conjugate Gradient Optimisation." << endl;
  }
  int Dimension = getNumPars();
   lb.resize(1, Dimension);
  ub.resize(1, Dimension);
  lb.ones();
  lb = lb * 1e-4;
  ub = 6 * ub.ones(); 

  mat w(1, Dimension);
  mat rk(1, Dimension);
  mat pk(1, Dimension);
  
  double sigmak;// 0<sigmak <1
  double sigma =  1e-4;//      0< sigma <1e-4
  const double lambda_l = 10-3; //     0<lambda_l<1e-6
  double lambdaBar = 0.0;
  double fw = 0.0;
  double muk = 0.0;
  double alphak = 0.0;
  double betak;
  bool success = true;
  double Deltak;
  double lambda;

  get_GP_Pars(w);
  mat Xnew = w;
  fw = Grad_Values(rk);
  double fnew = fw;
  mat gnew = rk;
  mat rk_old = rk;
  mat g = gnew;
  mat gold = gnew;
  mat sk = gnew;
  double fw_old = fw;
  rk = -1*rk;
  pk = rk;
  double deltak = 0.0;
  if (verbose > 0)
    cout << "Iteration: " << iter << " -logL: " << fw << " Scale: " << lambda << endl;
  iter = 0;
  while(true)
  {
    iter++;
    // step 2
    // update sigma
    if(success)
    {
      // can get a divide by zero here if pp is too small.
    double div = sqrt(accu(pk * pk.t()));
    sigmak = sigma / div;
    
     Xnew = w + pk * sigmak;
      set_GP_Pars(Xnew);
      fnew = Grad_Values(gnew);
      sk = (gnew - gold) / sigma + lambda * pk; 
      deltak = accu(pk * sk.t());
      gold = gnew;
    }
    // Step 3
    double norm2p = (accu(pk * pk.t()));
    deltak += (lambda - lambdaBar) *norm2p;
    sk += (lambda-lambdaBar) *pk;    
    // Step 4
    if(deltak <= 0.0) // Make the Hessian matrix positive definite.
    {
      lambdaBar = 2.0*(lambda - deltak / norm2p);
      deltak -= lambda * norm2p; 
      lambda = lambdaBar;
    }
    
    // Step 5  Calculate step size.
    muk = accu( pk * rk.t() );
    alphak = muk / deltak;
    
    // Step 6 Calculate the comparison parameter.
    Xnew = w + alphak * pk;
    bool violate = ChkBndStat(Xnew, lb, ub);
    while (violate)
    {
      alphak /= 1.2;
      Xnew = w + alphak * pk;
      violate = ChkBndStat(Xnew, lb, ub);
            if ( alphak < epsilon )
      {
	Xnew = w;
	alphak = 0.0;
	break;
      }
    }
    set_GP_Pars(Xnew);
    double falpha = ObjVal();
    Deltak = 2.0*deltak*(fw - falpha)/pow(muk, 2.0);
    
    // Step 7  Check for successful reduction in the error.
    if(Deltak >= 0.0) 
    {
      // update w
      w = Xnew;
      fw = falpha;
      Grad_Values(g);
      rk = -g;
      lambdaBar = 0; 
      success = true;
      // restart the algorithm
      if(iter % Dimension == 0) 
	pk = rk;
      else
      {	      
	betak = (accu(rk * rk.t()) - accu (rk * rk_old.t()) ) / muk;
	pk = rk + betak * pk;
      }
      
      // Reduce the scale parameter
      if(Deltak >= 0.75) 
	lambda *= 0.25;
    }
    else
    {
      set_GP_Pars(w);
      lambdaBar = lambda; 
      success = false; 
    }
    rk_old = rk;
    if ( (fabs(fw-fw_old) < getTolObjVal() && (iter % 3) == 0 & iter > 10))
    {
      return;
    }    
    // Step 8 Increase the scale parameter
    if(Deltak < 0.25) 
      lambda += (deltak * (1 - Deltak) / accu (pk * pk.t() ) );
    
    // Step 9 
    if ( all(vectorise(rk) == 0) ) 
      break;
    if (iter >= getMaxIters() )
      break;
    if (verbose > 0)
      cout << "Iteration: " << iter << " -logL: " << fw << " Scale: " << lambda << endl;
  }
  set_GP_Pars(w);
}
