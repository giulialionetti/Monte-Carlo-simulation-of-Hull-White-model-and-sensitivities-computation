# all:
# nvcc -O3 -arch=sm_70 --use_fast_math -lcurand main.cu -o hw

# run: all
#./hw

#clean:
#rm -f hw


NVCC = nvcc
FLAGS = -O3 --use_fast_math -arch=sm_70 -lcurand
INC = -I include
SRC = src
BIN = bin

all: q1 q2 q3

q1: $(SRC)/q1.cu
	@mkdir -p $(BIN) data
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

q2: $(SRC)/q2.cu
	@mkdir -p $(BIN)
	$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

#q3: $(SRC)/q3_sensitivities.cu
 #@mkdir -p $(BIN) data
#$(NVCC) $(FLAGS) $(INC) $< -o $(BIN)/$@

# Run targets
run-q1: main
	CUDA_VISIBLE_DEVICES=3 ./$(BIN)/q1

run-q2a: q2
	./$(BIN)/q2


#run-q3: q3
#CUDA_VISIBLE_DEVICES=3 ./$(BIN)/q3

# Run all in sequence
run-all: run-q1 run-q2 #run-q2b run-q3

clean:
	rm -rf $(BIN) data/*.bin

.PHONY: all clean run-q1 run-q2 # run-q2b run-q3 run-all