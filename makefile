CXX = g++
CXXFLAGS = -std=c++20 -O3 -Wall
INCLUDES = -I include -I onnx-proto

PROTO_FLAGS = $(shell pkg-config --cflags --libs protobuf)
LDFLAGS = -Wl,-rpath,$(CONDA_PREFIX)/lib

TARGET = parser

SRCS = src/main.cpp onnx-proto/onnx-ml.pb.cc

$(TARGET): $(SRCS)
	$(CXX) $(SRCS) $(CXXFLAGS) $(INCLUDES) $(PROTO_FLAGS) $(LDFLAGS) -o build/$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: test

test: $(TARGET)
	./build/$(TARGET)
	@echo "--- Running Python Verification ---"
	python3 scripts/test.py

