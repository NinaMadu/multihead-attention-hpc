#include <stdio.h>

#include "serial/attention.h"
#include "openmp/attention.h"
#include "mpi/attention.h"

#define MAX 100

int main()
{
    int n;
    int choice;

    float matrix[MAX][MAX];

    printf("Enter matrix size: ");
    scanf("%d",&n);

    printf("Enter matrix values:\n");

    for(int i=0;i<n;i++)
        for(int j=0;j<n;j++)
            scanf("%f",&matrix[i][j]);

    printf("\nSelect Implementation\n");
    printf("1 - Serial\n");
    printf("2 - OpenMP\n");
    printf("3 - MPI\n");

    scanf("%d",&choice);

    switch(choice)
    {
        case 1:
            serial_attention(matrix,n);
            break;

        case 2:
            openmp_attention(matrix,n);
            break;

        case 3:
            mpi_attention(matrix,n);
            break;

        default:
            printf("Invalid choice\n");
    }

    return 0;
}