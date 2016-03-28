GENERATE_SUBSETS_SOURCE=generate_subsets.cpp
GENERATE_SUBSETS_BINARY=generate_subsets.o

all: $(GENERATE_SUBSETS_BINARY)


$(GENERATE_SUBSETS_BINARY): $(GENERATE_SUBSETS_SOURCE)
	g++ $(GENERATE_SUBSETS_SOURCE) --std=c++11 -O2 -o $(GENERATE_SUBSETS_BINARY)

