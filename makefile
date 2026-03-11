CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall
INCLUDES = -I include -I onnx-proto

PROTO_FLAGS = $(shell pkg-config --cflags --libs protobuf)
LDFLAGS = -Wl,-rpath,$(CONDA_PREFIX)/lib

TARGET = parser

SRCS = src/main.cpp onnx-proto/onnx-ml.pb.cc

.PHONY: clean

$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(CXXFLAGS) $(INCLUDES) $(PROTO_FLAGS) $(LDFLAGS) -o build/$(TARGET)

clean:
	rm -rf build/*
	find models -mindepth 1 ! -name 'squeezenet1.1-7.onnx' -delete

.PHONY: test

test: $(TARGET)
	@echo "--- Running Steelix Compiler ---"
	./build/$(TARGET)
	@echo "--- Running Numerical Verification & benchmark---"
	python3 scripts/verify-steelix.py
	python3 scripts/benchmark-steelix.py

