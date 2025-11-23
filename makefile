NVCC = nvcc
FLAGS = -O3 --use_fast_math -arch=sm_70 -lcurand
INC = -I include
SRC = src
BIN = bin

all: q1 q2

q1: $(SRC)/q1.cu
	@mkdir -p $(BIN) data
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

q2: $(SRC)/q2.cu
	@mkdir -p $(BIN) data
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

# Run targets
run-q1: q1
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q1

run-q2: q2
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q2

# Run all in sequence
run-all: run-q1 run-q2

clean:
	rm -rf $(BIN) data/*.bin

.PHONY: all clean run-q1 run-q2 run-all