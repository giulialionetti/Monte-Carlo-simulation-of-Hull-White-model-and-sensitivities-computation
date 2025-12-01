NVCC = nvcc
FLAGS = -O3 --use_fast_math -arch=sm_70 -lcurand
INC = -I include
SRC = src
BIN = bin

all: q1 q2 q3

q1: $(SRC)/1_bond_pricing.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

q2: $(SRC)/2_option_pricing.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

q3: $(SRC)/3_sensitivity_analysis.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

benchmark: $(SRC)/benchmark_reductions.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

run-q1: q1
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q1

run-q2: q2
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q2

run-q3: q3
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q3

run-benchmark: benchmark
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/benchmark

run-all: run-q1 run-q2 run-q3

analyze: run-all
	python3 analyze.py

clean:
	rm -rf $(BIN) data/*.json data/*.csv data/summary.txt plots/*.png

clean-all:
	rm -rf $(BIN) data plots

.PHONY: all clean clean-all run-q1 run-q2 run-q3 run-all analyze