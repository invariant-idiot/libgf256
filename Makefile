# Compiler and common flags
CC       ?= clang
AR       ?= ar

CFLAGS   ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -Werror
CPPFLAGS ?= -I.
LDFLAGS  ?=
LDLIBS   ?= -pthread

# Project directories
BUILD := build
SRC   := src
TESTS := tests
BENCH := bench

# Library sources
COMMON_SRC := \
	$(SRC)/gf256.c \
	$(SRC)/gf256_base.c \
	$(SRC)/gf256_cpu.c \
	$(SRC)/gf256_threading.c

SIMD_SRC := \
	$(SRC)/gf256_gfni_avx256.c \
	$(SRC)/gf256_gfni_avx512.c

# Convert src/foo.c -> build/foo.o
COMMON_OBJ := $(COMMON_SRC:$(SRC)/%.c=$(BUILD)/%.o)
SIMD_OBJ   := $(SIMD_SRC:$(SRC)/%.c=$(BUILD)/%.o)
OBJECTS    := $(COMMON_OBJ) $(SIMD_OBJ)

LIB := $(BUILD)/libgf256.a
TEST := $(BUILD)/test_gf256
BENCHMARK := $(BUILD)/bench_all

.PHONY: all clean test bench

# Default target
all: $(LIB)

# Create the build directory when needed.
$(BUILD):
	mkdir -p $@

# Common source files.
$(BUILD)/%.o: $(SRC)/%.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# AVX2 + GFNI implementation.
$(BUILD)/gf256_gfni_avx256.o: $(SRC)/gf256_gfni_avx256.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		-mavx2 -mgfni \
		-c $< -o $@

# AVX-512 + GFNI implementation.
$(BUILD)/gf256_gfni_avx512.o: $(SRC)/gf256_gfni_avx512.c | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		-mavx512f -mavx512bw -mgfni \
		-c $< -o $@

# Build the static library.
$(LIB): $(OBJECTS)
	$(AR) rcs $@ $^

# Build and run tests.
$(TEST): $(TESTS)/test_gf256.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		$< $(LIB) $(LDFLAGS) $(LDLIBS) \
		-o $@

test: $(TEST)
	./$(TEST)

# Build and run benchmarks.
$(BENCHMARK): $(BENCH)/bench_all.c $(LIB) | $(BUILD)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		$< $(LIB) $(LDFLAGS) $(LDLIBS) \
		-o $@

bench: $(BENCHMARK)
	./$(BENCHMARK)

# Remove all generated files.
clean:
	rm -rf $(BUILD)
