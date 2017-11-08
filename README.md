
# Gaussian Process with Symmetric Standardization and Anisotropic Kernel


//////////////////////////////////////////////////////////
Gaussian process with symmetric standardization and anisotropic kernel for
ore grad estimation. (C++ code) 
//////////////////////////////////////////////////////////
Bahram Jafrasteh
Ph.D. candidate
Isfahan University of Technology, Isfahan, Iran.
b.jafrasteh@gmail.com
//////////////////////////////////////////////////////////
-Requirements:
	-Armadillo
	-GNUPlot
//////////////////////////////////////////////////////////	
The software has been written with KDevelop on Ubuntu.
//////////////////////////////////////////////////////////
-How to compile using make file:
	$ make gp_ss_ak
//////////////////////////////////////////////////////////
Input file should be comma delimited or tab delimited.
-Example:
         -training
           ./gp_ss_ak -v 3 -pm 1 train -k ExpAns -kn 1 -o LBFGS train.txt train_model
         -testing
           ./gp_ss_ak -v 3 -pm 1 test test.txt train_model train.txt
//////////////////////////////////////////////////////////
Help commands:
	$./gp -h
	$./gp_ss_ak train -h
	$./gp_ss_ak test -h
//////////////////////////////////////////////////////////

