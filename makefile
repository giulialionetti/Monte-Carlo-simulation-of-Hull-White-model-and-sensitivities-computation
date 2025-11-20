all:
	nvcc -O3 -arch=sm_60 -lcurand main.cu -o hw

run: all
	./hw

clean:
	rm -f hw