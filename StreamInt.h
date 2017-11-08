// // // // Bahram Jafrasteh // // // //
// // // // Ph.D. candidate // // // //
// // // // Isfahan University of Technology, Isfahan, Iran. // // // //
// // // // b.jafrasteh@gmail.com // // // //
// // // // October 27, 2017 // // // //
#ifndef StreamIntfc_H
#define StreamIntfc_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include<vector>
#include <stdlib.h>
#include <sstream>
#include <stdio.h>
#include <iostream> 
#include <fstream>
#include <string>
#include <math.h>
#include <limits>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <climits>
#include <cfloat>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cfloat>
#include "Control.h"



using namespace std;


class StreamIntfce 
{
public:
  virtual ~StreamIntfce() {}
  virtual void StrmOut(ostream& out) const
  {
    ToFile_GP_Params(out);
  }
  

  static int ReadIntStrm(istream& in, const std::string fieldName)
  {
    string str = ReadStrStrm(in, fieldName);
    return atol(str.c_str());
  }
  static double ReadDoubleStrm(istream& in, const std::string fieldName)
  {
    string str = ReadStrStrm(in, fieldName);
    return atof(str.c_str());
  }
  static bool ReadBoolStrm(istream& in, const std::string fieldName)
  {
    string str = ReadStrStrm(in, fieldName);
    if(atol(str.c_str())!=0)
      return true;
    else
      return false;
  }
  
  static string ReadStrStrm(istream& in, const std::string fieldName)
  {
    string line;
    vector<string> tokens;
    getline(in, line);
    string cm = line.substr(0,1);
    while ( cm.compare("#") == 0 )
    {
      getline(in, line);
      cm = line.substr(0,1);
    }
    size_t pos = line.find("="); 
    string  Str = line.substr (pos+1);
    return Str;
  }

  
  
  virtual void StrmIn(istream& in) 
  {
    FromFile_GP_Params(in);
  }
  
  
  virtual void ToFile_GP_Params(ostream& out) const=0;
  virtual void FromFile_GP_Params(istream& in)=0;
  
  void WFile(const string fileName, const string comment="") const
  {
    ofstream out(fileName.c_str());
    if(!out)
    {cout <<"The file "<< fileName <<" is open.\n";
    exit(1);}
      out << comment << endl;
    StrmOut(out);
    out.close();
  }
  
  void RFile(const string fileName) 
  {
    ifstream in(fileName.c_str());
    if(!in.is_open()) 
    {
      cout << "The file could not be read. \n";
      exit(1);
    }
    StrmIn(in);
    in.close();
  }
};

#endif /* StreamIntfc_H*/
