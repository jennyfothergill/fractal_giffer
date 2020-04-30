#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>

__global__ void allocate_2dcuda(
    int N, int M, int ghost, int *qmem, int** qrows, int*** q
)
{
    int rows = N + 2*ghost;
    int cols = M + 2*ghost;

    for(int i = 0; i < rows; i++)
    {
        qrows[i] = &qmem[cols*i + ghost];
    }
    *q = &qrows[ghost];
}

#define MAX_SHARED_MEMORY 1000

__global__ void julia_update(
    int N, int M, int ghost, int kmax, double rho, double creal, 
    double cimag, int ***dev_escape, double ax, double ay, double dx, double dy
)
{
    int **escape;
    escape = *dev_escape;
    
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    if(i < N && j < M)
    {
        double zreal = ax + i*dx + dx/2;
        double zimag = ay + j*dy + dy/2;
        cuDoubleComplex c = make_cuDoubleComplex(creal,cimag);
        cuDoubleComplex z = make_cuDoubleComplex(zreal,zimag);
        
        for(int k = 0; k < kmax; k++)
        {
            if(cuCabs(z) > rho && escape[i][j] == 0)
            {
                escape[i][j] = k; 
            }
            else
            {
                z = cuCadd(cuCmul(z,z), c);
            }
        }
    }
}
