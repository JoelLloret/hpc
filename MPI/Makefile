
CC = mpicc
CFLAGS = -fopenmp -O2 -Wall -std=c99
LDFLAGS = -lm

SRC = heat.c
EXE = heat

all: $(EXE)

$(EXE): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(EXE) *.o *.bmp
