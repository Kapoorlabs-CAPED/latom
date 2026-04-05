CXX = g++
CXXFLAGS = -std=c++17 -O3 -funroll-all-loops -finline-functions -ffast-math -Wall -Wno-deprecated
LDFLAGS = -lm

SOURCES = TDSE.cc wavefunction.cc hamop.cc grid.cc fluid.cc
OBJECTS = $(SOURCES:.cc=.o)
TARGET = TDSE

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean
