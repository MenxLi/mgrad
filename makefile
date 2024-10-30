
CXX_FLAGS = -std=c++17 -Wall -Isrc

LIB_STEMS = nn_graph nn_ops
OBJS = $(addprefix bin/, $(addsuffix .o, $(LIB_STEMS)))

.PHONY: test

bin:
	mkdir -p bin

bin/%.o: src/%.cc bin
	g++ $(CXX_FLAGS) -c $< -o $@

test: $(OBJS)
	g++ $(CXX_FLAGS) test/t0.cc $(OBJS) -o bin/test0
	g++ $(CXX_FLAGS) test/t1.cc $(OBJS) -o bin/test1

test-run: test
	@echo "----- Running tests -----"
	@./bin/test0 && ./bin/test1
