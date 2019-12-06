PROTBUFLIB=-L./lib -lprotobuf 
PROTOBUFINCLUDE=-I./include/external/protobuf_archive/src
TFINCULDE=-I./include  
TFLIBS=-L./lib -ltensorflow_cc -ltensorflow_framework
ACTLROOT=/home/ctauser/software/CamerasToACTL/trunk
ZFITSLIBS=-L$(ACTLROOT)/Build.Yves/lib -lACTLCore -lZFitsIO
ZFITSINCLUDE=-I$(ACTLROOT)/IO/Fits -I$(ACTLROOT)/Core -I$(ACTLROOT)/Build.Yves/Core
ROOTLIBS=$(shell root-config --libs)
ROOTINCLUDE=-I$(shell root-config --incdir)

CPP=g++
CPPFLAGS=-std=c++11 -O3 -g -Wall
LDFLAGS=-Wl,-rpath=./lib

all: convert_zfits

convert_zfits: convert_zfits.o zfits_datafile.o root_datafile.o
	$(CPP) -o $@ $^ $(LDFLAGS) $(ROOTLIBS) $(ZFITSLIBS) $(PROTBUFLIB)

convert_zfits.o: convert_zfits.cpp root_datafile.h root_datafile-impl.cpp
	$(CPP) -c -o $@ $< $(CPPFLAGS) $(ROOTINCLUDE) $(ZFITSINCLUDE) $(PROTOBUFINCLUDE)

zfits_datafile.o: zfits_datafile.cpp zfits_datafile.h root_datafile.h root_datafile-impl.cpp
	$(CPP) -c -o $@ $< $(CPPFLAGS) $(ROOTINCLUDE) $(ZFITSINCLUDE) $(PROTOBUFINCLUDE)

root_datafile.o: root_datafile.cpp root_datafile.h root_datafile-impl.cpp statistics.h statistics-impl.cpp
	$(CPP) -c -o $@ $< $(CPPFLAGS) $(ROOTINCLUDE)

.PHONY: clean all

clean:
	rm -f *.o *.a *~

