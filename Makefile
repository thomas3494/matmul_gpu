C = nvcc
FLAGS = -lineinfo -O3 -arch=native #-DNDEBUG
LFLAGS = -lm -lcublas
SRCS = $(wildcard src/*.cu)
BINS = $(patsubst src/%.cu, bin/%, $(SRCS))

all: $(BINS)

bin/%: src/%.cu include/test_code.h include/debug.h
	$(C) -Iinclude $(FLAGS) $< -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
