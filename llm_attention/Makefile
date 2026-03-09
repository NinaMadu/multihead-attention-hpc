# ── Makefile ──────────────────────────────────────────────────────────────────
# Builds the GPT-2 serial vs parallel attention benchmark.
#
# Targets:
#   make            – build optimised binary (release)
#   make debug      – build with -g -O0 (no optimisation, for gdb)
#   make run        – build + run with default prompt
#   make bench      – build + run benchmark (5 runs)
#   make clean      – remove build artefacts
#
# Override the number of OpenMP threads at runtime:
#   OMP_NUM_THREADS=4 ./llm_attn "Hello world"
# ──────────────────────────────────────────────────────────────────────────────

CC      = gcc
SRCDIR  = src
SRCS    = $(SRCDIR)/main.c          \
          $(SRCDIR)/attention_serial.c  \
          $(SRCDIR)/attention_parallel.c \
          $(SRCDIR)/forward.c        \
          $(SRCDIR)/weights_io.c     \
          $(SRCDIR)/tokenizer.c

TARGET  = llm_attn

# ── Release flags ─────────────────────────────────────────────────────────────
CFLAGS_REL  = -O3 -march=native -funroll-loops -ffast-math
CFLAGS_BASE = -Wall -Wextra -std=c11 -I$(SRCDIR)
LDFLAGS     = -lm -fopenmp

# OpenMP flag
OMP_FLAG = -fopenmp

# ── Default: release build ────────────────────────────────────────────────────
all: $(TARGET)

$(TARGET): $(SRCS) $(SRCDIR)/attention.h
	$(CC) $(CFLAGS_BASE) $(CFLAGS_REL) $(OMP_FLAG) -o $@ $(SRCS) $(LDFLAGS)
	@echo ""
	@echo "Build successful → ./$(TARGET)"
	@echo "Run:  ./$(TARGET) \"Hi how are you?\""
	@echo ""

# ── Debug build ───────────────────────────────────────────────────────────────
debug: $(SRCS) $(SRCDIR)/attention.h
	$(CC) $(CFLAGS_BASE) -O0 -g $(OMP_FLAG) -o $(TARGET)_debug $(SRCS) $(LDFLAGS)
	@echo "Debug build → ./$(TARGET)_debug"

# ── Quick run ─────────────────────────────────────────────────────────────────
run: all
	./$(TARGET) -g 20 "Hi how are you?"

# ── Benchmark (5 runs, both serial & parallel) ────────────────────────────────
bench: all
	./$(TARGET) -r 5 -g 5 "Hi how are you?"

# ── Serial-only run ───────────────────────────────────────────────────────────
serial: all
	./$(TARGET) -s -g 20 "Hi how are you?"

# ── Parallel-only run ─────────────────────────────────────────────────────────
parallel: all
	OMP_NUM_THREADS=4 ./$(TARGET) -p -g 20 "Hi how are you?"

# ── Thread sweep benchmark ────────────────────────────────────────────────────
sweep: all
	@echo "Thread sweep benchmark:"
	@for t in 1 2 4 8; do \
	    printf "  Threads=$$t: "; \
	    OMP_NUM_THREADS=$$t ./$(TARGET) -r 3 -g 1 "Hi" 2>/dev/null | \
	        grep "Parallel attention" | awk '{print $$4, $$5}'; \
	done

clean:
	rm -f $(TARGET) $(TARGET)_debug

.PHONY: all debug run bench serial parallel sweep clean
