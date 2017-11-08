#
#  Top Level Makefile for GP-SS-AK
#  Version 0.5
#  October 25, 2017


include make_linux

all: gp_ss_ak gp_ss_ak_lib$(LIBSEXT) gp_ss_ak_lib$(LIBDEXT)

gp_ss_ak: gp_ss_ak.o Control.o GP_Utils.o Opt_pars.o Kernel.o
	$(LD) ${XLINKERFLAGS} -o gp_ss_ak gp_ss_ak.o GP_Utils.o Control.o Opt_pars.o Kernel.o $(LDFLAGS)

gp_ss_ak_lib$(LIBSEXT): Control.o GP_Utils.o Opt_pars.o Kernel.o
	$(LIBSCOMMAND) gp_ss_ak_lib$(LIBSEXT) GP_Utils.o Control.o Opt_pars.o Kernel.o

gp_ss_ak_lib$(LIBDEXT): Control.o GP_Utils.o Opt_pars.o Kernel.o
	$(LIBDCOMMAND) gp_ss_ak_lib$(LIBDEXT) GP_Utils.o Control.o Opt_pars.o Kernel.o

gp_ss_ak.o: gp_ss_ak.cpp gp_ss_ak.h  \
  StreamInt.h Kernel.h \
  ModelInf.h GP_Utils.h Opt_pars.h Control.h
	$(CC) -c gp_ss_ak.cpp -o gp_ss_ak.o $(CCFLAGS)

Control.o: Control.cpp Control.h \
   StreamInt.h
	$(CC) -c Control.cpp -o Control.o $(CCFLAGS)


Kernel.o: Kernel.cpp Kernel.h \
  StreamInt.h \
  ModelInf.h
	$(CC) -c Kernel.cpp -o Kernel.o $(CCFLAGS)


Opt_pars.o: Opt_pars.cpp Opt_pars.h  \
   StreamInt.h
	$(CC) -c Opt_pars.cpp -o Opt_pars.o $(CCFLAGS)



GP_Utils.o: GP_Utils.cpp GP_Utils.h \
   Opt_pars.h StreamInt.h \
  Kernel.h ModelInf.h
	$(CC) -c GP_Utils.cpp -o GP_Utils.o $(CCFLAGS)

clean:
	rm *.o
