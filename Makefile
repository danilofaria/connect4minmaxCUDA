build: 
	nvcc c4.cu -o c4 -arch=sm_30

test: build 
	./c4 input.txt

.PHONY: exec 
exec: build
