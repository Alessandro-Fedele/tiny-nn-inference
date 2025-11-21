# Simple Makefile for Tiny Neural Inference Engine (TNIE)

CC      = gcc
CFLAGS  = -Wall -Wextra -std=c11 -O2 -Iinclude
LDFLAGS = -lm      # math library for expf()

SRC_DIR = src
SRCS    = $(SRC_DIR)/main.c \
          $(SRC_DIR)/tnie_nn.c \
          $(SRC_DIR)/tnie_activations.c \
          $(SRC_DIR)/tnie_model_xor.c

OBJS    = $(SRCS:.c=.o)
TARGET  = tnie_xor_demo

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TARGET)
