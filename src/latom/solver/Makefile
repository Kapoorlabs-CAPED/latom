CXX = g++
CXXFLAGS = -std=c++17 -O3 -funroll-all-loops -finline-functions -ffast-math -Wall -Wno-deprecated
LDFLAGS = -lm

SHARED_SOURCES = wavefunction.cc hamop.cc grid.cc fluid.cc
SHARED_OBJECTS = $(SHARED_SOURCES:.cc=.o)

TARGETS = TDSE ExactTDDFT

all: $(TARGETS)

TDSE: TDSE.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

ExactTDDFT: ExactTDDFT.o $(SHARED_OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o $(TARGETS)

.PHONY: all clean
