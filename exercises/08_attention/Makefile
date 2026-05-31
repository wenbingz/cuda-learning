NVCC ?= nvcc
TARGET := attention_naive
SRC := attention_naive.cu

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) -O2 -std=c++17 -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
