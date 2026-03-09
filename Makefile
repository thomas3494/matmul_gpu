C = nvcc
FLAGS = -O3 -DNDEBUG
LFLAGS = -lm -lcublas
SRCS = $(wildcard src/*.cu)
BINS = $(patsubst src/%.cu, bin/%, $(SRCS))

all: $(BINS)

bin/%_profile: src/%.cu include/test_code.h include/debug.h
	$(C) -Iinclude $(FLAGS) -lineinfo -DPROFILE $< -o $@ $(LFLAGS)

bin/%: src/%.cu include/test_code.h include/debug.h
	$(C) -Iinclude $(FLAGS) $< -o $@ $(LFLAGS)

clean:
	$(RM) bin/*
