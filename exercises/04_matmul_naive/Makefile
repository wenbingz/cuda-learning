NVCC ?= nvcc
TARGET := matmul_naive
SRC := matmul_naive.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

