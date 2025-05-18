
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

#define BMP_HEADER_SIZE 54
#define ALPHA 0.01
#define L 0.2
#define DX 0.02
#define DY 0.02
#define DT 0.0005
#define T 1500

int compute_local_rows(int rank, int base_nx, int rest) {
    return base_nx + (rank < rest ? 1 : 0);
}

void initialize_local_grid(double *grid, int local_nx, int ny, int global_start_row, int temp_source, int rank, int size) {
    for (int i = 0; i < local_nx - 1; i++) {
        for (int j = 0; j < ny; j++) {
            int global_i = global_start_row + i - 1;
            if (global_i == j || global_i == (ny - 1 - j)) {
                if (!((rank == 0 && global_i == 0 && (j == 0 || j == ny - 1)) ||
                      (rank == size - 1 && global_i == ny - 1 && (j == 0 || j == ny - 1)))) {
                    grid[i * ny + j] = temp_source;
                } else {
                    grid[i * ny + j] = 0.0;
                }
            } else {
                grid[i * ny + j] = 0.0;
            }
        }
    }
}


void exchange_halos(double *grid, int local_nx, int ny, int rank, int size)
{
    MPI_Request req[4];
    int nreq = 0;

    // local_nx -> number of rows in the local grid
    // ny -> number of columns in the local grid, and size of a row

    if (rank > 0) {
        // send upper interior to previous rank
        MPI_Isend(grid + ny, ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[nreq++]);
       
        // recv upper halo row from previous rank
        MPI_Irecv(grid, ny, MPI_DOUBLE, rank-1, 0, MPI_COMM_WORLD, &req[nreq++]);
    }
    
    if (rank < size-1) {
        // send lower interior to next rank
        MPI_Isend(grid + (local_nx-2)*ny, ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[nreq++]);
       
        // recv lower halo row from next rank
        MPI_Irecv(grid + (local_nx-1)*ny, ny, MPI_DOUBLE, rank+1, 0, MPI_COMM_WORLD, &req[nreq++]);
    }

    // wait for all send and receive operations to complete
    MPI_Waitall(nreq, req, MPI_STATUSES_IGNORE);
}

void solve_heat_equation(double *grid, double *new_grid, int steps, double r, int local_nx, int ny, int rank, int size) {
    int step, i, j;
    double *temp;

    for (step = 0; step < steps; step++) {
        
        exchange_halos(grid, local_nx, ny, rank, size);
        // OMP parallel region - dynamic scheduling for better load balancing
        #pragma omp parallel for private(j) schedule(dynamic)
        // ranks 0 and size-1 dont have to compute the first and last rows, as they don't have neighbors
        for (i = (rank == 0 ? 2 : 1); i < (rank == size - 1 ? local_nx - 2: local_nx - 1); i++) {
            for (j = 1; j < ny - 1; j++) {
            new_grid[i * ny + j] = grid[i * ny + j]
                + r * (grid[(i + 1) * ny + j] + grid[(i - 1) * ny + j] - 2 * grid[i * ny + j])
                + r * (grid[i * ny + j + 1] + grid[i * ny + j - 1] - 2 * grid[i * ny + j]);
            }
        }

        temp = grid;
        grid = new_grid;
        new_grid = temp;
    }
}

void write_bmp_header(FILE *file, int width, int height) {
    unsigned char header[BMP_HEADER_SIZE] = {0};
    int file_size = BMP_HEADER_SIZE + 3 * width * height;
    header[0] = 'B';
    header[1] = 'M';
    header[2] = file_size & 0xFF;
    header[3] = (file_size >> 8) & 0xFF;
    header[4] = (file_size >> 16) & 0xFF;
    header[5] = (file_size >> 24) & 0xFF;
    header[10] = BMP_HEADER_SIZE;
    header[14] = 40;
    header[18] = width & 0xFF;
    header[19] = (width >> 8) & 0xFF;
    header[20] = (width >> 16) & 0xFF;
    header[21] = (width >> 24) & 0xFF;
    header[22] = height & 0xFF;
    header[23] = (height >> 8) & 0xFF;
    header[24] = (height >> 16) & 0xFF;
    header[25] = (height >> 24) & 0xFF;
    header[26] = 1;
    header[28] = 24;
    fwrite(header, 1, BMP_HEADER_SIZE, file);
}

void get_color(double value, unsigned char *r, unsigned char *g, unsigned char *b) {
    if (value >= 500.0) { *r = 255; *g = 0; *b = 0; }
    else if (value >= 100.0) { *r = 255; *g = 128; *b = 0; }
    else if (value >= 50.0) { *r = 171; *g = 71; *b = 188; }
    else if (value >= 25) { *r = 255; *g = 255; *b = 0; }
    else if (value >= 1) { *r = 0; *g = 0; *b = 255; }
    else if (value >= 0.1) { *r = 5; *g = 248; *b = 252; }
    else { *r = 255; *g = 255; *b = 255; }
}

void write_grid(FILE *file, double *grid, int nx, int ny) {
    for (int i = nx - 1; i >= 0; i--) {
        for (int j = 0; j < ny; j++) {
            unsigned char r, g, b;
            get_color(grid[i * ny + j], &r, &g, &b);
            fwrite(&b, 1, 1, file);
            fwrite(&g, 1, 1, file);
            fwrite(&r, 1, 1, file);
        }
        for (int padding = 0; padding < (4 - (ny * 3) % 4) % 4; padding++) {
            fputc(0, file);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Usage: %s size steps output.bmp\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int nx = atoi(argv[1]), ny = nx;
    int steps = atoi(argv[2]);
    double r = ALPHA * DT / (DX * DY);
    int base_nx = nx / size, rest = nx % size;

    int local_rows = compute_local_rows(rank, base_nx, rest);
    int local_nx = local_rows + 2;
    int global_start_row = rank * base_nx + (rank < rest ? rank : rest);

    double *grid = calloc(local_nx * ny, sizeof(double));
    double *new_grid = calloc(local_nx * ny, sizeof(double));
    initialize_local_grid(grid, local_nx, ny, global_start_row, T, rank, size);

    printf("Rank %d - %d row\n", rank, global_start_row);

    double start, elapsed;
    start = MPI_Wtime();
    solve_heat_equation(grid, new_grid, steps, r, local_nx, ny, rank, size);
    double *final_grid = NULL;

    // rank 0 collects the results from all ranks and writes the output
    if (rank == 0) {
        final_grid = malloc(nx * ny * sizeof(double));
        memcpy(final_grid, grid + ny, local_rows * ny * sizeof(double));
        int offset = local_rows;
        for (int src = 1; src < size; src++) {
            int recv_rows = compute_local_rows(src, base_nx, rest);
            MPI_Recv(final_grid + offset * ny, recv_rows * ny, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += recv_rows;
        }
    // the other ranks send their results to rank 0
    } else {
        MPI_Send(grid + ny, local_rows * ny, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        FILE *file = fopen(argv[3], "wb");
        if (!file) {
            printf("Error opening the output file.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        write_bmp_header(file, nx, ny);
        write_grid(file, final_grid, nx, ny);
        fclose(file);
        free(final_grid);
    }

    free(grid);
    free(new_grid);

    elapsed = MPI_Wtime() - start;
    if (rank == 0) {
        printf("Execution Time = %f s with matrix size %d x %d and %d steps\n", elapsed, nx, ny, steps);
    }

    MPI_Finalize();
    return 0;
}
