#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>


#define MAX_SHARED_MEMORY 1000
__global__ void mandelbrot_update(
    int N, int M, int ghost, int kmax, double rho, int ***dev_escape, 
    double ax, double ay, double dx, double dy
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
        cuDoubleComplex z = make_cuDoubleComplex(zreal,zimag);
        cuDoubleComplex z0 = z;
        for(int k = 0; k < kmax; k++)
        {
            if(cuCabs(z) > rho && escape[i][j] == 0)
            {
                escape[i][j] = k; 
            }
            else
            {
                z = cuCadd(cuCmul(z,z), z0);
            }
        }
    }
}
