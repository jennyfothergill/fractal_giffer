#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h> 

#define IDX(i,j,cols,mbc)   i*cols + j + mbc

__global__ void allocate_2dcuda(
    int N, int M, int ghost, int *qmem, int **qrows, int ***q
);

__global__ void julia_update(
    int N, int M, int ghost, int kmax, double rho, double creal, double cimag, 
    int ***dev_escape, double ax, double ay, double dx, double dy
);

void allocate_2di(int N, int M, int mbc, int ***q)
{
    int rows = N + 2*mbc;
    int cols = M + 2*mbc; 

    int   *qmem = (int*)malloc(rows*cols*sizeof(int));
    int **qrows = (int**)malloc(rows*sizeof(int*));

    for(int i = 0; i < rows; i++)
    {
        qrows[i] = &qmem[cols*i + mbc];
    }    
    *q = &qrows[mbc];
}

void delete_2di(int mbc, int ***q)
{
    free(&(*q)[-mbc][-mbc]);
    free(&(*q)[-mbc]);
    *q = NULL;
}

int main(int argc, char** argv)
{
    /* ------------------------------ Input parameters ---------------------------- */
    /* ---  How to run -->  ./julia_cuda N kmax width xc yc creal cimag   --------- */ 

    int N        = atoi(argv[1]);
    int M        = N;
    int kmax     = atoi(argv[2]);
    double width = atof(argv[3]);
    double xc    = atof(argv[4]);
    double yc    = atof(argv[5]);
    double creal = atof(argv[6]);
    double cimag = atof(argv[7]);

    /* --------------------------- Numerical parameters --------------------------- */
    
    double ax = xc - width/2;
    double bx = xc + width/2;
    double ay = yc - width/2;
    double by = yc + width/2;
    double dx = (bx-ax)/N;
    double dy = (by-ay)/M; 

    /* ---------------------------- Initialize solution --------------------------- */

    int ghost = 0;

    int **escape;
    allocate_2di(N, M, ghost, &escape);

    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            escape[i][j] = 0;
        }
    }

    /* ---------------------------- Setup CUDA arrays ----------------------------- */
    
    int rows = M + 2*ghost;
    int cols = N + 2*ghost;
    
    int *dev_escape_mem, **dev_escape_rows, ***dev_escape;
    cudaMalloc( (void**) &dev_escape_mem, rows*cols*sizeof(int));
    cudaMalloc( (void***) &dev_escape_rows, rows*sizeof(int*));
    cudaMalloc( (void****) &dev_escape, sizeof(int**));
    
    allocate_2dcuda<<<1,1>>>(
        N, M, ghost, dev_escape_mem, dev_escape_rows, dev_escape
    );
    
    cudaMemcpy(
        dev_escape, &escape[-ghost][-ghost], (N*M)*sizeof(int), 
        cudaMemcpyHostToDevice
    );
    
    /* --------------------------- Start stepping ----------------------------------*/

    double rho = 2;
    
    int gx = 4;
    int gy = 4;
    dim3 block(gx, gy);
    dim3 grid((M+block.x - 1)/block.x, (N+block.y - 1)/block.y);

    /* ----- Time loop -- compute zk/escape at each step -----  */
    
    julia_update<<<grid,block>>>(
        N, M, ghost, kmax, rho, creal, cimag, dev_escape, ax, ay, dx, dy
    );
    
    cudaDeviceSynchronize();
    cudaMemcpy(
        &escape[-ghost][-ghost], dev_escape_mem, 
        (N*M)*sizeof(int), cudaMemcpyDeviceToHost
    );
    
    /* Write out meta data  */
    FILE *fout = fopen("julia_cuda.out","w");        
    fwrite(&N,1,sizeof(int),fout);
    fwrite(&M,1,sizeof(int),fout);
    fwrite(&ax,1,sizeof(double),fout);
    fwrite(&bx,1,sizeof(double),fout);
    fwrite(&ay,1,sizeof(double),fout);
    fwrite(&by,1,sizeof(double),fout);
    fwrite(&creal,1,sizeof(double),fout);
    fwrite(&cimag,1,sizeof(double),fout);
    
    /* --- (int) N , M, (double) ax, bx, ay, by, creal, cimag --- */
    
    /*----- write out escape -----*/
    fwrite(&escape[0][0],(N)*(M),sizeof(int),fout);
    
    fclose(fout);

    delete_2di(ghost,&escape);

    return 0;
}
