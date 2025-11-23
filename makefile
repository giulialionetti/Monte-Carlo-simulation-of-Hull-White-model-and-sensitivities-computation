NVCC = nvcc
FLAGS = -O3 --use_fast_math -arch=sm_70 -lcurand
INC = -I include
SRC = src
BIN = bin

all: q1 q2

q1: $(SRC)/q1.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

q2: $(SRC)/q2.cu
	@mkdir -p $(BIN) data plots
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

run-q1: q1
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q1

run-q2: q2
	@mkdir -p data plots
	CUDA_VISIBLE_DEVICES=2 ./$(BIN)/q2

run-all: run-q1 run-q2

analyze: run-all
	python3 analyze.py

clean:
	rm -rf $(BIN) data/*.json data/*.csv data/summary.txt plots/*.png

clean-all:
	rm -rf $(BIN) data plots

.PHONY: all clean clean-all run-q1 run-q2 run-all analyze