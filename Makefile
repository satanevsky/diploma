GENERATE_SUBSETS_SOURCE=generate_subsets.cpp
GENERATE_SUBSETS_BINARY=generate_subsets.so

all: $(GENERATE_SUBSETS_BINARY)


$(GENERATE_SUBSETS_BINARY): $(GENERATE_SUBSETS_SOURCE)
	g++ $(GENERATE_SUBSETS_SOURCE) --std=c++11 -o $(GENERATE_SUBSETS_BINARY) -shared -fPIC -O2 -lboost_system -lboost_python -lboost_numpy -I /usr/include/python2.7


