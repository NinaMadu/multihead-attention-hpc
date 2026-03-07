#include <stdio.h>
#include "attention.h"

void serial_attention(float matrix[100][100], int n)
{
    float result[100][100];

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            result[i][j] = 0;

            for(int k=0;k<n;k++)
                result[i][j] += matrix[i][k] * matrix[k][j];
        }
    }

    printf("\nSerial Result:\n");

    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
            printf("%.2f ", result[i][j]);

        printf("\n");
    }
}