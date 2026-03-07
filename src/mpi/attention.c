#include <stdio.h>
#include <mpi.h>
#include "attention.h"

void mpi_attention(float matrix[100][100], int n)
{
    int rank, size;

    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    float result[100][100];

    int rows = n / size;
    int start = rank * rows;
    int end = start + rows;

    for(int i=start;i<end;i++)
    {
        for(int j=0;j<n;j++)
        {
            result[i][j] = 0;

            for(int k=0;k<n;k++)
                result[i][j] += matrix[i][k] * matrix[k][j];
        }
    }

    MPI_Gather(result[start], rows*n, MPI_FLOAT,
               result, rows*n, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if(rank == 0)
    {
        printf("\nMPI Result:\n");

        for(int i=0;i<n;i++)
        {
            for(int j=0;j<n;j++)
                printf("%.2f ", result[i][j]);

            printf("\n");
        }
    }

    MPI_Finalize();
}