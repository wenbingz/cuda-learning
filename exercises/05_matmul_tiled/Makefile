NVCC ?= nvcc
TARGET := matmul_tiled
SRC := matmul_tiled.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

