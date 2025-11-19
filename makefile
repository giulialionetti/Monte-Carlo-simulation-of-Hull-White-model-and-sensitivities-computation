all:
	nvcc -O3 -arch=sm_70 -lcurand main.cu -o hw

run: all
	./hw

clean:
	rm -f hw