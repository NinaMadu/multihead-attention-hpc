CC = gcc
CFLAGS = -Wall -std=c99 -O2
LIBS = -lm

SERIAL_SRC = src/serial/attention.c src/main.c
SERIAL_OUT = serial_attention

all: $(SERIAL_OUT)

$(SERIAL_OUT): $(SERIAL_SRC)
	$(CC) $(CFLAGS) -o $(SERIAL_OUT) $(SERIAL_SRC) $(LIBS)

run: $(SERIAL_OUT)
	./$(SERIAL_OUT)

clean:
	rm -f $(SERIAL_OUT)

.PHONY: all run clean
