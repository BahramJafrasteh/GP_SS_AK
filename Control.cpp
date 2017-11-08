

#include "Control.h"

Control::Control(int arc, char** arv) : argc(arc), argv(arv)
{
  argNo = 1;
  prepareM = 1; // preparation method
  verbose = 0;
  time_t seconds;
  time(&seconds);
}

bool Control::isArg(string shortName, string Comnd)
{
  return (getArg() == shortName || getArg() == Comnd);
}

void Control::UnkFlg()
{
  ErrorTermination("Unknown flag: " + getArg() + " provided.");
}




void Control::readDataFile(mat &X, mat &y, int *data_size, const string InputFileName)
{
  ifstream in(InputFileName.c_str());
  if (!in.is_open())
    ErrorTermination("File is " + InputFileName + " not readable");
  string line;
  string token;
  int InputDimension;
  bool InputR = false;
  int numData = data_size[0];
  int MaxD = data_size[1];
  ifstream inToo(InputFileName.c_str());
  int pointNo = 0;
  while (getline(inToo, line))
  {
    InputR = false;
    if (line[0] == '#')
      continue;
    else
    {
      int position = 0;
      InputDimension = 0;
      while (position < line.size())
      {
	token.erase();
	while (position < line.size() && (line[position] != '\t' 
	  && line[position] != ','))
	{
	  token += line[position];
	  position++;
	}
	position++;
	if (token.size() > 0)
	{
	  if (InputDimension < MaxD)
	  {  
	    string StrInp = token.substr(0, position); 
	    if (InputDimension > MaxD || pointNo < 0 || pointNo >= numData)
	    {
	      ErrorTermination("Erro while reading" + InputFileName);
	    }
	    double Val = atof(StrInp.c_str());
	    
	    X(pointNo, InputDimension) = Val;
	    InputDimension ++;
	  }
	  else
	  {
	    string StrInp = token.substr(0, position);
	    y[pointNo] = atof(StrInp.c_str());
	  }
	}
      }
      
    }
    pointNo++;
  }
  in.close(); 
}

int* Control::readDataSize(const string InputFileName)
{
  int *data_size = new int[2];
  ifstream in(InputFileName.c_str());
  if (!in.is_open())
    ErrorTermination("File is " + InputFileName + " not readable");
  string line;
  string token;
  int InputDimension;
  bool InputR = false;
  int numData = 0;
  int MaxD = 0;
  while (getline(in, line))
  {
    InputR = false;
    if (line[0] == '#')
      continue;
    numData++;
    int position = 0;
    InputDimension = 0;
    while (position < line.size())
    {
      token.erase();
      while (position < line.size() && (line[position] != '\t' 
	&& line[position] != ','))
      { 
	token += line[position];
	position++;
      }
      position++;
      
      if (token.size() > 0)
      {
	if (InputR)
	{  
	  InputDimension ++;
	  if (InputDimension > MaxD)
	    MaxD = InputDimension;
	}
	else
	{
	  InputR = true;
	}
      }
    }
    
  }
  data_size[1] = MaxD;
  data_size[0] = numData;
  if (verbose > 0){
    cout << "Number of features in the input file are: " << MaxD << endl;
    cout << "Number of readable data are: " << numData << endl;}
    in.close();
    return (data_size);
}
void Control::prepareData(mat& X, mat& y, int &Data_mode, bool &yscale, string ModelN)
{
  params.zeros(X.n_cols + y.n_cols, 2);
  MinData.resize(y.n_cols+X.n_cols, 1);
  MaxData.resize(y.n_cols+X.n_cols, 1);
  MeanData.resize(y.n_cols+X.n_cols, 1);
  StData.resize(y.n_cols+X.n_cols, 1);
  if (getMode() == "train")
    StatisticsCalc(X, y);
  else if (getMode() == "test"){
    mat Statistics;
    Statistics.load(ModelN+"_Statistics.txt", csv_ascii);
    params = Statistics.cols(0, 1);
    MinData = Statistics.col(2);
    MaxData = Statistics.col(3);
    MeanData = Statistics.col(4);
    StData = Statistics.col(5);
    MaxTotalin = MaxData.submat(1, 0, X.n_cols, 0).max();
    MaxTotalo = MaxData[0];
    MinTotalin = MinData.submat(1, 0, X.n_cols, 0).min();
    MinTotalo = MinData[0];
  }
  switch (prepareM)
  { 
    case 0:// Mean and Std
      MeanStd(X, y, Data_mode, yscale);
      if (verbose > 0)
	cout<<"Preparation method is between mean and standardDev and y scale is "<< yscale<< endl;
      break;
    case 1:// symmetric preparation
      prep_symmetric(X, y, Data_mode, yscale);
      if (verbose > 0)
	cout<<"Preparation method is symmetric and y scale is "<< yscale<< endl;
      break;
    case 2:// zeroandone
      zeroandone(X, y, Data_mode, yscale);
      if (verbose > 0)
	cout<<"Preparation method is between 0 and 1 and y scale is "<< yscale<< endl;
      break;
    default:
    {
      ErrorTermination("Unrecognised preparation method.");
    }
  }
  
  if (getMode() == "train")
  {
    mat Statistics = join_horiz( params, MinData );
    Statistics = join_horiz( Statistics, MaxData );
    Statistics = join_horiz( Statistics, MeanData );
    Statistics = join_horiz( Statistics, StData );
    Statistics.save(ModelN+"_Statistics.txt", csv_ascii);
  }
}

void Control::postData(mat& X, mat& y, bool &yscale, string ModelN)
{
  if ( getMode() == "test")
  {   mat Statistics;
    Statistics.load(ModelN+"_Statistics.txt", csv_ascii);
    params = Statistics.cols(0, 1);
    MinData = Statistics.col(2);
    MaxData = Statistics.col(3);
    MeanData = Statistics.col(4);
    StData = Statistics.col(5);
    MaxTotalin = MaxData.submat(1, 0, X.n_cols, 0).max();
    MaxTotalo = MaxData[0];
    MinTotalin = MinData.submat(1, 0, X.n_cols, 0).min();
    MinTotalo = MinData[0];
  }
  //   mat d = params.submat(y.n_cols, 0, X.n_cols+ y.n_cols-1, 0).t();
  for (int j = 0; j < X.n_cols; j++)
  {
    X.col(j) = (X.col(j) * params(j+1, 1)) + params(j+1, 0) ;
  }
  if (yscale)
    y = (y * params(0, 1)) + params(0, 0) ;
}

void Control::postData(mat& X, bool &yscale, string ModelN)
{
  if ( getMode() == "test")
  {   mat Statistics;
    Statistics.load(ModelN+"_Statistics.txt", csv_ascii);
    params = Statistics.cols(0, 1);
    MinData = Statistics.col(2);
    MaxData = Statistics.col(3);
    MeanData = Statistics.col(4);
    StData = Statistics.col(5);
    MaxTotalin = MaxData.submat(1, 0, X.n_cols, 0).max();
    MaxTotalo = MaxData[0];
    MinTotalin = MinData.submat(1, 0, X.n_cols, 0).min();
    MinTotalo = MinData[0];
  }
  X = X* params(0, 1) + params(0, 0);
}
void Control::postData_var(mat& X, bool &yscale, string ModelN)
{
  if ( getMode() == "test")
  {   mat Statistics;
    Statistics.load(ModelN+"_Statistics.txt", csv_ascii);
    params = Statistics.cols(0, 1);
    MinData = Statistics.col(2);
    MaxData = Statistics.col(3);
    MeanData = Statistics.col(4);
    StData = Statistics.col(5);
    MaxTotalin = MaxData.submat(1, 0, X.n_cols, 0).max();
    MaxTotalo = MaxData[0];
    MinTotalin = MinData.submat(1, 0, X.n_cols, 0).min();
    MinTotalo = MinData[0];
  }
  if (yscale)
    X = sqrt(X*pow(params(0,1), 2));
}

void Control::MeanStd(mat& X, mat& y, int &Data_mode, bool &yscale)
{
  
  if (getMode() == "train")
  {
    params(0, 0) = MeanData[0];
    params(0, 1) = StData[0];
    for (int j = 0; j < X.n_cols; j++)
    {
      params(j+1, 0) = MeanData[j+1] ;
      params(j+1,1) = StData[j+1];
    }
  }
  for (int j = 0; j < X.n_cols; j++)
  {
    X.col(j) = (X.col(j) - params(j+1, 0)) / params(j+1, 1);
  }
  if (yscale)
    y = (y - params(0, 0)) / params(0, 1);
}

void Control::zeroandone(mat& X, mat& y, int &Data_mode, bool &yscale)
{
  params(0, 0)= 0.5 * (MinData[0]);
  params(0, 1) = 0.5 * (MaxData[0] - MinData[0]);
  for (int j = 0; j < X.n_cols; j++)
  {
    params(j+1, 0) = 0.5 * (MinData[j+1]);
    params(j+1, 1) = 0.5 * (MaxData[j+1] - MinData[j+1]);
  }
  for (int i = 0; i < X.n_rows; i++)
  {
    for (int j = 0; j < X.n_cols; j++)
    {
      X(i, j) = (X(i, j) - params(j+1, 0)) / params(j+1, 1);
    }
    if (yscale)
      y[i] = (y[i] - params(0, 0)) / params(0, 1);
  }
}


void Control::prep_symmetric(mat& X, mat& y, int &Data_mode, bool &yscale)
{
  if (getMode() == "train")
  {
    
    params(0, 0) = 0.5 * (MaxTotalo + MinTotalo);
    params(0, 1) = 0.5 * (MaxTotalo - MinTotalo);
    for (int j = 0; j < 3; j++)
    {
      params(j+1, 0) = 0.5 * (MaxTotalin + MinTotalin);
      params(j+1,1) = 0.5 * (MaxTotalin - MinTotalin);
    }      
    for (int j = 3; j < X.n_cols; j++)
    {
      params(j+1, 0) = 0.5 * (MaxData[j+1] + MinData[j+1]);
      params(j+1, 1) = 0.5 * (MaxData[j+1] - MinData[j+1]);
    }
  }
  
  for (int j = 0; j < X.n_cols; j++)
  {
    X.col(j) = (X.col(j) - params(j+1, 0)) / params(j+1, 1);
  }
  if (yscale)
    y = (y - params(0, 0)) / params(0, 1);
}


  void Control::Helping()
  {
    cout << "To get more information use help command." << endl;
  }
  void Control::ErrorTermination(const string error)
  {
    
    cerr << error << endl << endl;
    Helping();
    exit(1);
  }