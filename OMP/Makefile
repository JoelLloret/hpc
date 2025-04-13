# Compiler
CC=gcc
SOURCE=heat_omp.c
EXEC=output

# Flags default
OPTIONS= -lm -fopenmp

all:
	$(CC) $(SOURCE) $(OPTIONS) -o $(EXEC)
clean:
	rm $(EXEC)