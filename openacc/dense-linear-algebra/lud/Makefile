# === Makefile for LUD (OpenACC) ===

CXX = nvc++

BASEFLAGS = -std=c++17 -O3 -Wall -Wextra -g
ACCFLAGS  = -acc -gpu=cc75 -Minfo=accel

CXXFLAGS = $(BASEFLAGS) $(ACCFLAGS)

SRC = main.cpp lud_serial.cpp lud_parallel.cpp
OBJ = $(SRC:.cpp=.o)
BIN = lud

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(BIN)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

clean:
	rm -f $(OBJ) $(BIN) lud_perf.csv