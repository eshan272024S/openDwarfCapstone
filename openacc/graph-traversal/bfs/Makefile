CXX = nvc++
CXXFLAGS = -O3 -std=c++17 -acc -Minfo=accel

LIBS      ?=

SRCS = main.cpp bfs_serial.cpp bfs_openacc.cpp
OBJS = $(SRCS:.cpp=.o)
BIN = bfs_acc

all: $(BIN)

$(BIN): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run-c: $(BIN)
	./$(BIN) -c

run-p: $(BIN)
	./$(BIN) -p

run-both: $(BIN)
	./$(BIN) -c -p

download:
	bash scripts/get_data.sh

clean:
	rm -f $(OBJS) $(BIN)

.PHONY: all run-c run-p run-both download clean

