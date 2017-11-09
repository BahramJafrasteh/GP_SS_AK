// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //



//   Main File
#include "gp_ss_ak.h"
#include <stdlib.h>
using namespace arma;

int main(int argc, char* argv[])
{
  
  GP_Cntrl Comnd(argc, argv);
  Comnd.setFlgs(true);
  Comnd.setprepM(1);// 0: -1 and 1, 1: symmetric, 2: 0 and 1
  Comnd.setVerbose(0);
  Comnd.setMode("gp");
  
  while (Comnd.isFlgs())
  {
    string argm = Comnd.getArg();
    if (argv[Comnd.getArgNo()][0] == '-')
    {
      if (Comnd.isArg("-?", "--?"))
      {
	Comnd.Help();
	exit(0);
      }
      else if (Comnd.isArg("-h", "--help"))
      {
	Comnd.Help();
	exit(0);
      }
      else if (Comnd.isArg("-v", "--verboseL"))
      {
	Comnd.incArg();
	Comnd.setVerbose(Comnd.getIntArg());
      }
      else if (Comnd.isArg("-pm", "--prepMethod"))
      {
	Comnd.incArg();
	Comnd.setprepM(Comnd.getIntArg());
      }
      else
	Comnd.UnkFlg();
    }
    else if (argm == "train") // learning a model.
      Comnd.train();
    else if(argm=="test") // test with a model.
      Comnd.test();
    else
      Comnd.ErrorTermination("Invalid Commad.");
    Comnd.incArg();
  }
  Comnd.ErrorTermination("No Command.");
  
  
  
}
GP_Cntrl::GP_Cntrl(int arc, char** arv) : Control(arc, arv)
{
}
void GP_Cntrl::train()
{
  incArg();
  setMode("train");
  bool yscale = true;
  double tol = 1e-6;
  string optimiser = "BFGS";
  int Data_mode = 0; //  0 for training data
  int numhyper; // number of hyperparameters
  vector<string> KernT;
  int likeLtype = -1;
  int Inf_type = -1;
  int mean_type = -1;
  int Ker_t = 0;
  bool Knoise = "true";
  string mean_functionstr = "mean_zero";
  string likelihoodf = "Gauss";
  string Inference = "inf_laplace";
  int numMF_par;
  int numlik_par;
  //     vector<int> labels;
  uvec labels;
  int iters = 100;
  string modelName = "gp_model";
  while (isFlgs()) {
    if (isArgFlg())
    {
      if (isArg("-h", "--help")) {
	Help();
	exit(0);
      }
      else if (isArg("-mf", "--meanfunction")) {
	incArg();
	mean_functionstr = getArg();
      }
      else if (isArg("-lf", "--likefunction")) {
	incArg();
	likelihoodf = getArg();
      }
      else if (isArg("-k", "--kernel")) {
	
	incArg();
	KernT.push_back(getArg());
	Ker_t++;
      }
      else if (isArg("-o", "--optimiser")) {
	incArg();
	optimiser = getArg();
      }
      else if (isArg("-#", "--iterations")) {
	incArg();
	iters = getIntArg();
      }
      else if (isArg("-kn", "--Knoise")) {
	incArg();
	Knoise = getIntArg();
      }
      else {
	UnkFlg();
      }
      incArg();
    }
    else
      setFlgs(false);
  }
  if (getArgNo() >= argc)
    ErrorTermination("There are not enough input parameters.");
  string trainFileName = getArg();
  if ((getArgNo() + 1) < argc)
    modelName = argv[getArgNo() + 1];
  string data_File = "";
  int *data_size = new int[2];
  data_size = readDataSize(trainFileName);
  mat X(data_size[0], data_size[1]);
  mat y(data_size[0], 1);
  readDataFile(X, y, data_size, trainFileName); 
  prepareData(X, y, Data_mode, yscale, modelName);
  int inputDim = X.n_cols;    
  // create covariance function.
  HybKerns Kerns(X);
  vector<Kernels*> kernels;
  for (int i = 0; i < KernT.size(); i++) {
    mat M = X;
    
    if (KernT[i] == "RBF") {
      kernels.push_back(new Kern_RBF(M));
    }
    else if (KernT[i] == "ExpAns") {
      
      kernels.push_back(new Kern_ExpAnisotropic(M));
      
    }
    else if (KernT[i] == "Exp") {
      
      kernels.push_back(new Kern_Exponential(M));
    }
    else if (KernT[i] == "Bias") {
      // bias component
      kernels.push_back(new Kern_Bias(M));
    }
    else if (KernT[i] == "White") {
      // white noise component
      kernels.push_back(new Kern_White(M));
    }
    else {
      ErrorTermination("Unknown covariance function: " + KernT[i]);
    }
    Kerns.addNewKernel(kernels[i]);
    
  }
  // if no covariance function was specified, add a Kernel.
  if (Kerns.getNumKerns() == 0)
  {
    Kernels* defaultKern = new Kern_ExpAnisotropic(X);
    Kerns.addNewKernel(defaultKern);
    KernT.push_back(getArg());
    KernT[0] = "ExpAn";
  }
  if (Knoise){
    Kernels* biasKern = new Kern_Bias(X);
    Kerns.addNewKernel(biasKern);
    //       Kernels* whiteKern = new CWhiteKern(X);
    //       kern.addKern(whiteKern);
  }

  if (likelihoodf == "Gauss")
  {
    likeLtype = GP_utils::likeL_Gaussian;
    numlik_par = 1;
  }
  
  
  if (Inference == "inf_laplace")
  {
    Inf_type = GP_utils::inf_laplace;
  }
  if (mean_functionstr == "mean_zero")
  {
    mean_type = GP_utils::mean_zero;
  }
  
  
  // Number of hyperparameters
  if (KernT[0] == "RBF")
    numhyper = 2;
  else if (KernT[0] == "ExpAn")
    numhyper = 8;
  else if (KernT[0] == "Exp")
    numhyper = 2;
  else if (KernT[0] == "Bias")
    numhyper = 2;
  else if (KernT[0] == "White")
    numhyper = 2;
  
  // number of mean function parameters
  if (mean_functionstr == "mean_zero")
    numMF_par = 0;   
  else{
    ErrorTermination("Unrecognised mean function");
  }
  

  
  GP_utils* GPModel;
  GPModel = new GP_utils(&Kerns, X, y, 
			 Inf_type, likeLtype, mean_type, numhyper, 
			 numlik_par, numMF_par, getVerbose());

  cout << "The inital value of the kernel parameters are as follows :" 
  << endl;
  cout << "There are " << GPModel->KerenlW->getNPars() << " parameters to be optimized"
  << endl;  
  for (int i = 0; i < GPModel->KerenlW->getNPars(); i++)
    cout << GPModel->KerenlW->getParamName(i)<< " : " << GPModel->KerenlW->getParam(i)<<endl;
  cout << "Do you want to change the defult kernel parameters (Yes|Y|y or press any key)?" << endl;
  string res = "No";
   cin >> res;
  if ( res == "Yes" || res == "Y" || res == "y" )
  {
    for (int i = 0; i < GPModel->KerenlW->getNPars(); i++)
    {
      
      if ( GPModel->KerenlW->getParamName(i) == "InversewidthR_ExpAns" &&
	X.n_cols == 3)
	continue;
      cout << " Please input an initial value for " << GPModel->KerenlW->getParamName(i) << 
	      " (Default value was " << GPModel->KerenlW->getParam(i)
      << ") : " <<endl;
      long double d;
      d = GPModel->KerenlW->getParam(i);
      if (cin.peek() != '\n')
	cin >> d;
      cin.ignore(numeric_limits<streamsize>::max(), '\n');
      GPModel->KerenlW->setParam(d, i);    
    }
  }





  cout << "The inital value of the likelihood function are as follows :"<< endl;
  cout << "likelihood hyperparameter : " << GPModel->getHyperlfVal(0) <<endl;
    cout << "Do you want to change the defult likelihood function parameters (Yes|Y|y or press any key)?" << endl;
  res = "No";
   cin >> res;
  if ( res == "Yes" || res == "Y" || res == "y" )
  {
    cout << "Please input an initial value for Gauss likelihood function" 
    " : " <<endl;
    long double d;
    d = GPModel->getHyperlfVal(0);
    if (cin.peek() != '\n')
      cin >> d;
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
    GPModel->setHyperlfVal(d, 0);
  }
  else{
  }
  if (optimiser == "SCG")
    GPModel->setOptimiser(GP_utils::SCG);
  else if (optimiser == "LBFGS")
    GPModel->setOptimiser(GP_utils::LBFGS);
  else if (optimiser == "BFGS")
    GPModel->setOptimiser(GP_utils::BFGS);
  else
    ErrorTermination("Unrecognised optimiser type: " + optimiser);
  
  GPModel->OptimisePars(iters);
  
  
  string comment = "# GP_SS_AK Model File ";
  writeGPFile(*GPModel, modelName, comment);
  
  mat EstVals(X.n_rows, GPModel->getOutDim());
  mat EstVals_Var(X.n_rows, GPModel->getOutDim());
  GPModel->Calc_Out(EstVals, EstVals_Var, X);
  postData(X, EstVals, yscale, modelName);
  postData_var(EstVals_Var, yscale, modelName);
  postData(y, yscale, modelName);
  EstVals_Var = sqrt(EstVals_Var);
  //       uvec ind_neg = find(EstVals < 0.0);
  //       EstVals.elem(ind_neg).fill(0.0);
  bool status = y.has_nan();
  bool statusn = EstVals.has_nan();
  double mse = accu( pow(y - EstVals, 2) ) / X.n_rows;
  mat dY = y - accu(y)/y.n_rows;
  dY = pow(dY, 2);
  double varMSE_tr = accu(dY)/ y.n_rows;
  if (getVerbose() > 0)
  {
    cout << "Mean Square Error of training: " << mse << "\n";
    cout << "Var MSE Train: " << varMSE_tr << "\n";
  }
  else
  {
    cout << mse << "\n";
    cout << varMSE_tr << "\n";
  }
  exit(0);
}




void GP_Cntrl::test()
{
  incArg();
  setMode("test");
  double pointSize = 2;
  double lineWidth = 2;
  int Data_mode = 1; //  1 for testing data
  bool yscale = true;
  string name = "gp_ss_ak";
  string data_File_NameTr;
  string modelName = "model";
  while (isFlgs()) {
    if (isArgFlg()) {
      int j = 1;
      if (getArgLen() != 2)
	UnkFlg();
      else if (isArg("-?", "--?") || isArg("-h", "--help")) {
	Help();
	exit(0);
      }
      else {
	UnkFlg();
      }
      incArg();
    }
    else
      setFlgs(false);
  }
  if (getArgNo() >= argc)
    ErrorTermination("There are not enough input parameters.");
  string data_File_Name = getArg();
  if ((getArgNo() + 1) < argc)
    modelName = argv[getArgNo() + 1];
  if ((getArgNo() + 2) < argc)
    data_File_NameTr = argv[getArgNo() + 2];
  else
    cout << "Please provide training data \n";
  string outputFileName = name + "_plot_data";
  string PredictOut =  modelName + "_predict.txt";
  if ((getArgNo() + 3) < argc)
  {
    outputFileName = argv[getArgNo() + 3];
    PredictOut = outputFileName;
  }
  int *data_size = new int[2];
  data_size = readDataSize(data_File_Name);
  mat X(data_size[0], data_size[1]);
  mat y(data_size[0], 1);
  readDataFile(X, y, data_size, data_File_Name); 
  prepareData(X, y, Data_mode, yscale, modelName);
  GP_utils* GPModel = readGpFromFile(modelName, getVerbose());
  
  data_size = readDataSize(data_File_NameTr);
  mat Xtr(data_size[0], data_size[1]);
  mat ytr(data_size[0], 1);    
  readDataFile(Xtr, ytr, data_size, data_File_NameTr); 
  prepareData(Xtr, ytr, Data_mode, yscale, modelName);
  GPModel->yTarg = ytr;
  GPModel->Xinp = Xtr;
  int numdata = Xtr.n_rows;
  int numout = ytr.n_cols;
  GPModel->setNumData(Xtr.n_rows);
  GPModel->initialize_vars();
  GPModel->logLikelihood();
  
  /// START HERE
  if (X.n_cols != GPModel->getInpDim()) {
    ErrorTermination("Incorrect dimension of input data.");
  }
  
  
  
  
  int j;
  int i;
  mat EstVals(y.n_rows, y.n_cols);
  mat EstVals_Var(y.n_rows, y.n_cols);
  GPModel->Calc_Out(EstVals, EstVals_Var, X);
  postData(X, EstVals, yscale, modelName);
  postData_var(EstVals_Var, yscale, modelName);
  postData(y, yscale, modelName);
  //       uvec ind_neg = find(EstVals < 0.0);
  //       EstVals.elem(ind_neg).fill(0.0);
  bool status = y.has_nan();
  bool statusn = EstVals.has_nan();
  double mse = accu( pow(y - EstVals, 2) ) / X.n_rows;
  mat dY = y - accu(y)/y.n_rows;
  dY = pow(dY, 2);
  double varMSE_t = accu(dY)/ y.n_rows;
  if (getVerbose() > 0)
  {
    cout << "Mean Square Error of testing: " << mse << "\n";
    cout << "Var MSE Test: " << varMSE_t << "\n";
  }
  else
  {
    cout << mse << "\n";
    cout << varMSE_t << "\n";
  }
  
  
  
  mat Rgsp = regspace<mat>(1, 1, y.n_rows);
  uvec indices = sort_index(y, "ascend");
  y = y.elem(indices);
  EstVals =  EstVals.elem(indices);
  EstVals_Var = EstVals_Var.elem(indices);
  for (int i = 0; i < X.n_cols; i++)
  {
    mat tmp = X.col(i);
    X.col(i) = tmp.elem(indices);
  }
  mat regr = join_horiz<mat>(Rgsp, y);
  regr = join_horiz<mat>(regr, EstVals);
  regr = join_horiz<mat>(regr, EstVals_Var);
  regr = join_horiz<mat>(regr, X);
  
  string tempN = data_File_Name;
  int ind = tempN.find("/");
  vector<string> parts;
  if (ind > 0)
  {
    while (ind > 0)
    {
      if (ind+1 < tempN.size())
      {
	parts.push_back(tempN.substr(0, ind));
	tempN = tempN.substr(ind+1, tempN.size());
      }
      ind = tempN.find("/");
    }
  }
  
  if(tempN.find("train"))
    modelName =  modelName + "_train";
  if(tempN.find("test"))
    modelName =  modelName + "_test";
  
  
  ofstream outputs(PredictOut.c_str());
  outputs<< "# SampleNo, Y,  Yh, StdYh, Inputs"<< "\n";
  for (int i = 0; i < regr.n_rows; i++)
  {
    for (int j = 0; j < regr.n_cols; j++)
    {
      outputs<< regr(i, j) << "\t";
    }
    outputs<< "\n";
  }
  outputs.close();
  string matF = modelName + "_gnu.plt";
  ofstream GNUout(matF.c_str());
  GNUout << "#gnuplot -persist output.plt\n set termoption enhanced\n set term wxt background rgb \"white\"\n set term pdf transparent enhanced \n ";
  string GNUoutFile = "set output '" +  modelName+"_predict.pdf" +"'  \n";
  GNUout << GNUoutFile;
  GNUout <<"set style fill transparent solid 0.70 noborder\n set grid nopolar\n set key inside left top vertical Right noreverse enhanced autotitle box lt black linewidth 1.000 dashtype solid\n set title \"Observed vs Estimated\" textcolor  \"black\" font \"Bold-Times-Roman,20\"\n";
  string DataName = "Copper grade";
  string P0 = "set ylabel \"" + DataName +"\" offset 0.1,0.1 textcolor  \"black\" font \"Bold-Times-Roman,10\" \n";
  GNUout << P0;
  GNUout << "set xlabel \"Sample\" offset 0.1,0.1 textcolor  \"black\" font \"Bold-Times-Roman,10\" \n";
  GNUout <<"set colorbox vertical origin screen 0.9, 0.2, 0 size screen 0.05, 0.6, 0 front bdefault \n";
  
  double maxT = max(y.max(), (EstVals + EstVals_Var).max());
  double minT = min(y.min(), (EstVals - EstVals_Var).min());
  
  string P1 = "plot [0.9:" + to_string((double)y.n_rows+0.01) +"] [" + to_string(minT-0.02)+":"+ to_string(maxT+0.02)+
  "] \"" + PredictOut + "\" using 1:($3 + $4):($3 - $4) with filledcurve fc rgb \"green\" title '95% CI', \"\" using 1:3 with lines ls 2 lw 1 lc rgb \"red\" t \"Estimated\", \"\" using 1:2 ls 1 lw 1 lc rgb \"blue\" t \"Observed\" with lines \n";
  GNUout<< P1;
  GNUout.close();
  string GNUcmd = "gnuplot -persist " + modelName + "_gnu.plt";
  char *GNUP=new char[GNUcmd.size()+1];
  GNUP[GNUcmd.size()]=0;
  memcpy(GNUP,GNUcmd.c_str(),GNUcmd.size());
  system(GNUP);
  
  exit(0);
}


void GP_Cntrl::Help()
{
  string Comnd = getMode();
  if (Comnd == "gp") {
    cout << endl << "GP_SS_AK Code: Version 0.5" << endl;
    cout<<"Command:\n \t ./gp_ss_ak [options] Command [Comnd-options] modelName TrainDataFile.txt"
    << endl;
    cout << "Commands:" << endl;
    cout << "train :\n \t To find hyperparameter by maxmizing likelihood."<< endl;
    cout << "test :\n \t To estimate test data set and plot the results."<< endl;
    cout <<"To get more information about each command please type command with --h"
    <<endl;
    cout << endl;
    cout << "Options:" << endl;
    cout << "-?, -h, --help\n \t To get help on options"<< endl;
    cout << "-v, --verbose\n \t Verbosity level (default 0)."<< endl;
    cout << "-pm, --prepMethod\n \t preparation methos (between mean and std [0], symmetric [1], ...)"<<
    endl;
  }
  else if (Comnd == "train") {
    cout << endl << "GP_SS_AK Code: Version 0.5" << endl;
    cout <<"gp [options] learn example_file [model_file]"<<endl;
    cout <<"This file is for learning a data set with an GP. By default 100 iterations of scaled conjugate gradient are used."
    << endl;
    cout << "Arguments:" << endl;
    cout <<"-mf ,--meanfunction\n \t GP mean function name (Default: zero [mean_zero])"<<endl;      
    cout <<"-lf, --likefunction\n \t likelihood function name (default:  [Gauss]"<< endl;
    cout <<"-k, --kernel\n \t Kernel name (default:  Exponential Anisotropic [ExpAns], other options [RBF]: RBF, EXponential [Exp])"<< endl;
    cout <<"-o, --optimiser\n \t Optimization algorithm (default : BFGS [BFGS], other options: Scale Conjugate gradient [SCG], ... )"<< endl;
    cout <<"-kn, --Knoise\n \t Noise Kernel (default: true [1], other options false [0])"<< endl;
    cout <<"trainFileName\n \t File containg trainig data (comma delimitted or tab delimitted file)."<< endl;
    cout <<"modelName\n \t File to store the model."<< endl;
    cout<<"-#, --iterations\n \t Number of iterations for optimisation algorithm. Default is 1000."
    <<endl;
  }
  
  else if (Comnd == "test") {
    cout << endl << "GP_SS_AK Code: Version 0.5" << endl;
    cout <<"[options] gp test [test_file] [model_file] [train_file]"<< endl;
    cout <<"This Command computes test estimation and visualise results using gnuplot."<< endl;
    cout << "Arguments:" << endl;
    cout <<"-v, --verbose\n \t Verbosity level (default 0)."<< endl;
    cout <<"test_file\n \t The test data file (comma delimitted or tab delimitted file). if you are going to estimate new data set the last column is the observed value."<< endl;
    cout <<"model_file\n \t The model you want to use in estimation."<< endl;
    cout <<"train_file\n \t The file which has been used in training."<< endl;
  }
}
