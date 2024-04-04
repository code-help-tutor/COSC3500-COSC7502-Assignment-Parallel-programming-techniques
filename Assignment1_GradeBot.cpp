WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
// COSC3500 Matrix multiply assignment

// Defining this line will enable LAPACK/BLAS functions. The program won't work without them, but it can help debugging.
#define LAPACKBLAS_ENABLE

// Defining this will make calls to the BLAS library using the CBLAS interface, rather than the FORTRAN interface
// e.g. cblas_cgesvd rather than cgesvd
// #define CBLAS_ENABLE

//For debugging purposes, you can disable CPU, GPU, and/or MPI sections of the code by commenting out the ENABLE_x defines
#define ENABLE_CPU
#define ENABLE_GPU
#define ENABLE_MPI

//Alignment boundary for aligned mallocs (e.g. _mm_malloc). 64 bytes = cacheline
#define ALIGN 64

//misc. standard C/C++ libraries
#include <string>
#include <iostream>
#include <chrono>
#include <random>
#include <stdio.h>
#include <thread>

//Intel MKL Library
#include <mkl_lapack.h> //cgesvd, sgels
#ifdef CBLAS_ENABLE
#include <mkl_cblas.h> //cgemv, cgemm
#else                  // Fortran interface
#include <mkl_blas.h>  //cgemv, cgemm
#endif
#include <mkl.h>

//Assignment headers
#include "matrixMultiply.h"

#ifdef ENABLE_GPU
#include "matrixMultiplyGPU.cuh"
#endif

#ifdef ENABLE_MPI
#include "matrixMultiplyMPI.h"
#endif
#include <omp.h>

#ifdef ENABLE_GPU
//CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas.h>
#endif

//Some code for getting information abouth the CPU
#include "cpuInfo.h"

//The header file that defines the marking rubric
#include "rubric.h"

//The physical (or hyperthreaded) cores reported by the operating system
const int cpuCount = std::thread::hardware_concurrency();

//The number of threads to use for openMP and MKL
int threadCount = cpuCount;

//The MPI rank of this process
int mpiRank = 0;
//The total number of MPI processes
int mpiWorldSize = 1;

//Turn off some annoying warnings if you're using Microsoft
#ifdef _MSC_VER
// Microsoft compiler complains about functions like fopen
#define _CRT_SECURE_NO_WARNINGS
// Disable Arithmetic overflow errors (e.g. pointless warnings about adding two ints and then multiplying by a size_t)
#pragma warning(disable : 26451)
#endif

//The reference CPU matrix multiply using MKL library
void matrixMultiply0(int Nx, float* A, float* B, float* y, char transA)
{
    int Ny = Nx;
    if (Ny >= Nx)
    {
        // A is Nx x Ny
        // B is Ny x Nx
        // C is Nx x Nx

        int M = Nx; // Number of rows of A and C
        int N = Nx; // Number of cols of B and C
        int K = Ny; // Number of columns of A and number of rows of B

        int LDA = Ny; // A (LDA,M) : When 'C' (LDAxM)
        int LDB = Ny; // B (LDB,N) : When 'N' (MxN)
        int LDC = Nx; // C (LDC,N)

        float alpha = 1;
        float beta = 0;

#ifdef LAPACKBLAS_ENABLE
#ifdef CBLAS_ENABLE
        cblas_sgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasConjTrans, CBLAS_TRANSPOSE::CblasNoTrans, M, N, K, alpha, A, LDA, B, LDB, beta, y, LDC);
#else
        // const char transA = 'N';//C
        const char transB = 'N';
        sgemm(&transA, &transB, &M, &N, &K, (float*)&alpha, (float*)A, &LDA, (float*)B, &LDB, (float*)&beta, (float*)y, &LDC);
#endif
#endif
    }
    else
    {
        printf("Error : Attempted to find A*A' of a matrix that's wider than it is high. e.g. batchCount is larger than mode count\n");
    }
}

//The reference GPU matrix multiply using the CUDA BLAS library
void matrixMultiply0_GPU(int Nx, float* A, float* B, float* C, char transA)
{
#ifdef ENABLE_GPU
    int Ny = Nx;
    if (Ny >= Nx)
    {
        // A is Nx x Ny
        // B is Ny x Nx
        // C is Nx x Nx

        int M = Nx; // Number of rows of A and C
        int N = Nx; // Number of cols of B and C
        int K = Ny; // Number of columns of A and number of rows of B

        int LDA = Ny; // A (LDA,M) : When 'C' (LDAxM)
        int LDB = Ny; // B (LDB,N) : When 'N' (MxN)
        int LDC = Nx; // C (LDC,N)

        float alpha = 1;
        float beta = 0;

        const char transB = 'N';
        //sgemm(&transA, &transB, &M, &N, &K, (float*)&alpha, (float*)A, &LDA, (float*)B, &LDB, (float*)&beta, (float*)y, &LDC);
        cublasSgemm(transA, transB, M, N, K, alpha, A, LDA, B, LDB, beta, C, LDC);

    }
    else
    {
        printf("Error : Attempted to find A*A' of a matrix that's wider than it is high. e.g. batchCount is larger than mode count\n");
    }
#endif
}

//A routine that looks at the difference between two matrices. Use for checking answers against the reference solution.
float matrixCheck(int N, float* A0, float* A, float* Atemp, float& err)
{
    matrixMultiply0(N, A, A0, Atemp, 'C');

    //Sum of eigenvalues
    float sumEigs = 0;
    for (int i = 0; i < N; i++)
    {
        sumEigs += Atemp[i * N + i] * Atemp[i * N + i];
    }

    //sum of squared difference errors
    err = 0;
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float vA = A0[i * N + j];
            float vB = A[i * N + j];
            err += (vA - vB) * (vA - vB);
        }
    }

    return sumEigs;
}

void matrixMultiplyRef_GPU(int N, float* A, float* X, float* Y)
{
    matrixMultiply0_GPU(N, A, X, Y, 'N');
}

void matrixMultiplyRef(int N, float* A, float* X, float* Y)
{
    matrixMultiply0(N, A, X, Y, 'N');
}

//Routine for creating a set of random unitary matrices. We'll use unitary matrices so that we can multiply them together indefinitely without the scale of the numbers in the matrices ever blowing out.
void generateRandomUnitaries(int N, int matrixCount, float* M_0, float* M_U, float* work, int lwork)
{
    int info = 0; // NULL
    int m = N;
    int n = N;
    int lda = N;
    // float* work = WORK;

    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {
        const size_t matrixOffset = matrixIdx * N * N;
        float* Ain = &M_0[matrixOffset];
        float* Aout = &M_U[matrixOffset];
        memcpy(Aout, Ain, sizeof(float) * N * N);
        float* tau = Ain;
#ifdef LAPACKBLAS_ENABLE
        sgeqrf(&m, &n, Aout, &lda, tau, work, &lwork, &info);
        int k = n - 1;
        sorgqr(&n, &n, &k, Aout, &lda, reinterpret_cast<float*>(tau), reinterpret_cast<float*>(work), &n, &info);
#endif
    }
}

float matrixMultiplyTest_RefGPU(int matrixCount, int N, float* M_U, float* Ain, float* Aout, float* Aref, float& errDelta)
{
    double matrixMulRateRef = 0;
#ifdef ENABLE_GPU
    
    cudaDeviceSynchronize();

    auto StartTimeRef = std::chrono::high_resolution_clock::now();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {

        //We won't count the first one, because CUDA might be initialising the first time a cuBLAS call is run.
        if (matrixIdx == 1)
        {
            StartTimeRef = std::chrono::high_resolution_clock::now();

            cudaEventRecord(start, 0);
        }
        float* tempPtr = Ain;
        Ain = Aout;
        Aout = tempPtr;

        const size_t matrixOffset = matrixIdx * N * N;

        float* X = &M_U[matrixOffset];
        float* A = (matrixIdx == 0) ? X : Ain;
        float* C = Aout;

        // C = A * X;
        matrixMultiplyRef_GPU(N, A, X, C);
        cudaDeviceSynchronize();
       
    }
    float time = 0;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    auto FinishTimeRef = std::chrono::high_resolution_clock::now();
    auto TotalTimeRef = std::chrono::duration_cast<std::chrono::microseconds>(FinishTimeRef - StartTimeRef);

    //Using the C timer
   //matrixMulRateRef = 1e6 * (matrixCount-1) / (1.0 * TotalTimeRef.count());

    //Using the cuda timers
    matrixMulRateRef = 1e6 * (matrixCount - 1) / (1.0 * time*1000);

    //Allocate memory for pulling the matrix back from the GPU
    float* Aout_CPU = (float*)_mm_malloc(sizeof(float) * N * N, ALIGN);
    cudaMemcpy(Aout_CPU, Aout, N * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    float* Atemp_CPU = (float*)_mm_malloc(sizeof(float) * N * N, ALIGN);

    //Compares with the CPU reference (Aref) (if already calculated)
    errDelta = 0;
    matrixCheck(N, Aref, Aout_CPU, Atemp_CPU, errDelta);

    //Remember this result as the reference matrix
    memcpy(Aref, Aout_CPU, sizeof(float) * N * N);

    _mm_free(Aout_CPU);
    _mm_free(Atemp_CPU);
#endif

    return matrixMulRateRef;
}

float matrixMultiplyTest_Ref(int matrixCount, int N, float* M_U, float *Ain, float *Aout, float* Aref,  float &errDelta)
{
    // The reference calculation (using intel MKL)
    auto StartTimeRef = std::chrono::high_resolution_clock::now();
    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {
        float* tempPtr = Ain;
        Ain = Aout;
        Aout = tempPtr;

        const size_t matrixOffset = matrixIdx * N * N;

        float* X = &M_U[matrixOffset];
        float* A = (matrixIdx == 0) ? X : Ain;
        float* C = Aout;
        // C = A * X;
        matrixMultiplyRef(N, A, X, C);
        
    }

    auto FinishTimeRef = std::chrono::high_resolution_clock::now();
    auto TotalTimeRef = std::chrono::duration_cast<std::chrono::microseconds>(FinishTimeRef - StartTimeRef);

    memcpy(Aref, Aout, N * N * sizeof(float));

    double matrixMulRateRef = 1e6 * matrixCount / (1.0 * TotalTimeRef.count());
    errDelta = 0;

    return matrixMulRateRef;
}

float matrixMultiplyTest_GPU(int matrixCount, int N, float* M_U, float* Ain, float* Aout, float* Aref, float& errDelta, int *args, int argCount)
{
    double matrixMulRate = 0;
#ifdef ENABLE_GPU

    cudaDeviceSynchronize();

    // The reference calculation (using intel MKL)
    auto StartTimeRef = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {
        //We won't count the first one, in case CUDA is initialising something for the first run.
        if (matrixIdx == 1)
        {
            StartTimeRef = std::chrono::high_resolution_clock::now();
            cudaEventRecord(start, 0);
        }

        float* tempPtr = Ain;
        Ain = Aout;
        Aout = tempPtr;

        const size_t matrixOffset = matrixIdx * N * N;

        float* X = &M_U[matrixOffset];
        float* A = (matrixIdx == 0) ? X : Ain;
        float* C = Aout;

        // C = A * X;
        matrixMultiply_GPU(N, A, X, C,args,argCount);
        cudaDeviceSynchronize();
    }

    float time = 0;
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);

    auto FinishTimeRef = std::chrono::high_resolution_clock::now();
    auto TotalTimeRef = std::chrono::duration_cast<std::chrono::microseconds>(FinishTimeRef - StartTimeRef);

    //Using the C/C++ timer
    //matrixMulRate = 1e6 * (matrixCount - 1) / (1.0 * TotalTimeRef.count());

    //Using the CUDA timer
    matrixMulRate = 1e6 * (matrixCount - 1) / (1.0 * time*1000);


    //Pull result back to CPU host for error checking the matrix result
    float* Aout_CPU = (float*)_mm_malloc(sizeof(float) * N * N, ALIGN);
    cudaMemcpy(Aout_CPU, Aout, N * N * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    float* Atemp_CPU = (float*)_mm_malloc(sizeof(float) * N * N, ALIGN);

    //Compare the resulting matrix with the reference solution
    errDelta = 0;
    matrixCheck(N, Aref, Aout_CPU, Atemp_CPU, errDelta);

    _mm_free(Aout_CPU);
    _mm_free(Atemp_CPU);
#endif
    return matrixMulRate;
}


float matrixMultiplyTest_MPI(int matrixCount, int N, float* M_U, float* Ain, float* Aout, float* Aref, float& errDelta, int *args, int argCount)
{
    double matrixMulRate = 0;

#ifdef ENABLE_MPI
    //All MPI processes wait here to start...
    MPI_Barrier(MPI_COMM_WORLD);
    auto StartTime = std::chrono::high_resolution_clock::now();
    
    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {
        //Ignore the first one (in case there's some initialisation taking place)
        if (matrixIdx == 1)
        {
            MPI_Barrier(MPI_COMM_WORLD);
            StartTime = std::chrono::high_resolution_clock::now();
        }

        float* tempPtr = Ain;
        Ain = Aout;
        Aout = tempPtr;

        const size_t matrixOffset = matrixIdx * N * N;

        float* X = &M_U[matrixOffset];
        float* A = (matrixIdx == 0) ? X : Ain;
        float* C = Aout;
        // C = A * X;
        matrixMultiply_MPI(N, A, X, C, args,argCount);
    }

    auto FinishTime = std::chrono::high_resolution_clock::now();
    auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

     matrixMulRate = 1e6 * (matrixCount-1) / (1.0 * TotalTime.count());

    matrixCheck(N, Aout, Aref, Ain, errDelta);
#endif
    return matrixMulRate;
}

float matrixMultiplyTest_CPU(int matrixCount, int N, float* M_U, float* Ain, float* Aout, float* Aref, float &errDelta, int *args, int argCount)
{
    auto StartTime = std::chrono::high_resolution_clock::now();

    for (int matrixIdx = 0; matrixIdx < matrixCount; matrixIdx++)
    {
        float* tempPtr = Ain;
        Ain = Aout;
        Aout = tempPtr;

        const size_t matrixOffset = matrixIdx * N * N;

        float* X = &M_U[matrixOffset];
        float* A = (matrixIdx == 0) ? X : Ain;
        float* C = Aout;
        // C = A * X;
        matrixMultiply(N, A, X, C, args,argCount);
    }

    auto FinishTime = std::chrono::high_resolution_clock::now();
    auto TotalTime = std::chrono::duration_cast<std::chrono::microseconds>(FinishTime - StartTime);

    double matrixMulRate = 1e6 * matrixCount / (1.0 * TotalTime.count());

    matrixCheck(N, Aout, Aref, Ain, errDelta);

    return matrixMulRate;
}

int benchmark_GPUInit(float* M_U, float *M_U_GPU, int64_t workspaceElements)
{
#ifdef ENABLE_GPU
    CUdeviceptr memDst = (CUdeviceptr)M_U_GPU;
    float* memSrc = M_U;
    cudaMemcpyKind cpyKind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    const size_t memSize = workspaceElements * sizeof(float);
    cudaMemcpy((void*)memDst, (void*)memSrc, memSize, cpyKind);
#endif
    return 1;
}

int benchmark_GPU(int matrixCount, int N, float* M_U_GPU, float* Ain_GPU, float* Aout_GPU, float* Aref, int Nmax, FILE* fid, int *args, int argCount)
{
#ifdef ENABLE_GPU
	float errDelta = 1e9;
    int gpuCount = 0;
    int gpuIdx = 0;

    cudaDeviceProp gpuProp;
    cudaGetDevice(&gpuIdx);
    cudaGetDeviceCount(&gpuCount);
    cudaGetDeviceProperties(&gpuProp,gpuIdx);

    char idString[8192];
    sprintf(&idString[0],"GPU[%i,%i|%i/%i](%s)",mpiRank,mpiWorldSize,gpuIdx,gpuCount,gpuProp.name);

	const int trialCount = 10;
	double matrixMulRateRef = 0;
	for (int trialIdx = 0; trialIdx < trialCount; trialIdx++)
	{
		matrixMulRateRef = matrixMultiplyTest_RefGPU(matrixCount, N, (float*)M_U_GPU, (float*)Ain_GPU, (float*)Aout_GPU, (float*)Aref, errDelta);
	}

	cudaDeviceSynchronize();

		//Reset the matrix results
		cudaMemset(Ain_GPU, 0, sizeof(float) * N * N);
		cudaMemset(Aout_GPU, 0, sizeof(float) * N * N);

		double matrixMulRate = 0;

		for (int trialIdx = 0; trialIdx < trialCount; trialIdx++)
		{
			matrixMulRate = matrixMultiplyTest_GPU(matrixCount, N, M_U_GPU, Ain_GPU, Aout_GPU, Aref, errDelta, args,argCount);
		}

		float perfFactor = (matrixMulRateRef / matrixMulRate);

		float grade = getGrade(perfFactor, errDelta, &rubric[RUBRIC_GPU][0]);

		fprintf(stdout, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);
		fprintf(fid, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);

#endif
	return 1;
}

int benchmark_CPU(int matrixCount, int N, float *M_U, float *Ain, float*Aout, float *Aref, int Nmax, FILE* fid, int *args, int argCount)
{
    float errDelta = 0;
    cpuINFO cpuInfo;
    cpuInfoGet(&cpuInfo);

    char idString[2048];
    sprintf(&idString[0],"CPU[%i,%i|%i/%i](%s)",mpiRank,mpiWorldSize,threadCount,cpuCount,cpuInfo.brand);
    double matrixMulRateRef = matrixMultiplyTest_Ref(matrixCount, N, M_U, Ain, Aout, Aref, errDelta);

        //Reset the matrix results
        memset(Aout, 0, sizeof(float) * Nmax * Nmax);
        memset(Ain, 0, sizeof(float) * Nmax * Nmax);

        double matrixMulRate = matrixMultiplyTest_CPU(matrixCount, N, M_U, Ain, Aout, Aref, errDelta,args,argCount);

        float perfFactor = (matrixMulRateRef / matrixMulRate);

        float grade = getGrade(perfFactor, errDelta, &rubric[RUBRIC_CPU][0]);

        fprintf(stdout, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);
        fprintf(fid, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);

    return 1;
}

int benchmark_MPI(int matrixCount, int N, float* M_U, float* Ain, float* Aout, float* Aref, int Nmax, FILE* fid, int* args, int argCount)
{
#ifdef ENABLE_MPI

    int world_size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int my_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    char name[4096];
    int resultlen=0;
    MPI_Get_processor_name(&name[0],&resultlen);

    char idString[8192];
    sprintf(&idString[0],"MPI[%i,%i|%i/%i](%s)",mpiRank,mpiWorldSize,threadCount,cpuCount,name);

    float errDelta = 0;

    double matrixMulRateRef = matrixMultiplyTest_Ref(matrixCount, N, M_U, Ain, Aout, Aref, errDelta);

        //Reset the matrix results
        memset(Aout, 0, sizeof(float) * Nmax * Nmax);
        memset(Ain, 0, sizeof(float) * Nmax * Nmax);

        double matrixMulRate = matrixMultiplyTest_MPI(matrixCount, N, M_U, Ain, Aout, Aref, errDelta, args,argCount);

        float perfFactor = (matrixMulRateRef / matrixMulRate);

        float grade = getGrade(perfFactor, errDelta, &rubric[RUBRIC_MPI][0]);

        fprintf(stdout, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);
        fprintf(fid, "%s	%i	%3.3f	%3.3f	%3.3f	%3.3e	%2.2f\n", idString,N, matrixMulRateRef, matrixMulRate, perfFactor, errDelta, grade);
#endif
    return 1;
}



int main(int argc, char** argv)
{
    //Default matrix dimension
    int N = 2048;
   
    //Defines whether the CPU, GPU, and MPI benchmarks should be run
    int isEnabled[3] = { 1,1,1};

   //Reading in the command-lin arguments
   //first argument will the the executable path/name itself
    int *argsIn = 0;
    argsIn = (int*)_mm_malloc(sizeof(int)*argc,ALIGN);
    memset(argsIn,0,sizeof(int)*argc);

    //Read in the command-line arguments
    for (int argIdx=1;argIdx<argc;argIdx++)
    {
        int v = atoi(argv[argIdx]);
        argsIn[argIdx-1] = v;

        if (argIdx==1 && v>0)
        {
            N = v;
        }
        if (argIdx==2 && v>0)
        {
            threadCount = v;
        }

        if (argIdx==3)
        {
            isEnabled[RUBRIC_CPU] = v;
        }
        if (argIdx==4)
        {
            isEnabled[RUBRIC_GPU] = v;
        }
        if (argIdx==5)
        {
            isEnabled[RUBRIC_MPI] = v;
        }
    }

    int *args = 0;
    int argCount = 0;

    if (argc>6)
    {
        args = &argsIn[5];
        argCount = argc-6;
    }

    const int64_t Nmax = N;

    //The amount of memory to allocate for test matrices
    int64_t goalMemorySize = 1024*1024*1024;//Gb of test data
    goalMemorySize/=4;

    int64_t matrixCount = goalMemorySize/(Nmax*Nmax*sizeof(float));
    if (matrixCount<=2)
    {
        matrixCount = 2;
    }
    const int64_t workspaceElements = Nmax * Nmax * matrixCount;

    //Keep track of the total amount of allocated memory (most for debugging)
    size_t totalAllocatedMemory = 0;

    //Arrays that will be used to store matrices and results
    //The raw array of random numbers
    float* M_0 = (float*)_mm_malloc(workspaceElements * sizeof(float), ALIGN);
    totalAllocatedMemory += workspaceElements * sizeof(float);

    //Array of random unitary matrices (that we will multiply by in our calculation)
    float* M_U = (float*)_mm_malloc(workspaceElements * sizeof(float), ALIGN);
    totalAllocatedMemory += workspaceElements * sizeof(float);

    //Input matrix
    float* Ain = (float*)_mm_malloc(Nmax * Nmax * sizeof(float), ALIGN);
    totalAllocatedMemory += Nmax * Nmax * sizeof(float);

    //Output matrix
    float* Aout = (float*)_mm_malloc(Nmax * Nmax * sizeof(float), ALIGN);
    totalAllocatedMemory += Nmax * Nmax * sizeof(float);

    //The reference solution matrix (e.g. by MKL or CUDA BLAS)
    float* Aref = (float*)_mm_malloc(Nmax * Nmax * sizeof(float), ALIGN);
    totalAllocatedMemory += Nmax * Nmax * sizeof(float);

    //workspace memory for MKL matrix decompositions (used to generate unitary matrices)
    const int lwork = 5 * Nmax;
    float* work = (float*)_mm_malloc(workspaceElements * sizeof(float), ALIGN);
    totalAllocatedMemory += lwork * sizeof(float);

#ifdef ENABLE_GPU

    //GPU memory allocation
    CUdeviceptr M_U_GPU = 0;
    CUdeviceptr Ain_GPU = 0;
    CUdeviceptr Aout_GPU = 0;

    if (isEnabled[RUBRIC_GPU])
    {
        cudaError_t errCode = cudaMalloc((void**)&M_U_GPU, workspaceElements * sizeof(float));

        errCode = cudaMalloc((void**)&Ain_GPU, Nmax * Nmax * sizeof(float));
        errCode = cudaMalloc((void**)&Aout_GPU, Nmax * Nmax * sizeof(float));
    }
    
#endif

    std::mt19937 rng(4);
    std::normal_distribution<float> dist(-1.0f, 1.0f);

    // Generate a bunch of random numbers between +/-1
    for (size_t i = 0; i < workspaceElements; i++)
    {
        M_0[i] = dist(rng);
    }

    FILE* fidGPU = 0;
    FILE* fidCPU = 0;
    FILE* fidMPI = 0;

    int isMPIinited = 0;

#ifdef ENABLE_MPI
    if (isEnabled[RUBRIC_MPI])
    {
        if (!isMPIinited)
        {
            MPI_Init(&argc, &argv);
            isMPIinited = true;

            #ifdef ENABLE_MPI
                MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
                MPI_Comm_size(MPI_COMM_WORLD, &mpiWorldSize);
            #endif

        }
    }
#endif
    if (isEnabled[RUBRIC_CPU])
    {
        char filename[1024];
        sprintf(&filename[0],"COSC3500Assignment_CPU_%2.2i.txt",mpiRank);
        fidCPU = fopen(filename, "w");
    }
    if (isEnabled[RUBRIC_GPU])
    {
        char filename[1024];
        sprintf(&filename[0],"COSC3500Assignment_GPU_%2.2i.txt",mpiRank);
        fidGPU = fopen(filename, "w");
    }

    if (isEnabled[RUBRIC_MPI])
    {
        char filename[1024];
        sprintf(&filename[0],"COSC3500Assignment_MPI_%2.2i.txt",mpiRank);
        fidMPI = fopen(filename, "w");
    }

    if (mpiRank==ROOT_NODE)
    {
        fprintf(stdout, "Info.	N	Matrices/second (MKL)	Matrices/second (You)	You/MKL	error	Grade\n");
    }

    //Initialisation routines
    rubricInit();
    matrixCount = workspaceElements / (N * N);
    generateRandomUnitaries(N, matrixCount, M_0, M_U, work, lwork);
  
    omp_set_nested(1);
    omp_set_dynamic(0);
    //I wouldn't have thought I need this line because of the omp_set_num_threads call, but without it when using MPI it seems to default to 1 openMP thread
    //Might be this issue...
    //https://community.intel.com/t5/Intel-oneAPI-Math-Kernel-Library/Intel-MKL-2019-2021-no-longer-threads-internally-when-using-MPI/td-p/1250963
    omp_set_num_threads(threadCount);
    mkl_set_num_threads(threadCount);
        
    //Benchmark CPU performance
    if (isEnabled[RUBRIC_CPU])
    {
        benchmark_CPU(matrixCount, N, M_U, Ain, Aout, Aref, Nmax, fidCPU, args, argCount);
    }

    //Benchmark GPU performance
#ifdef ENABLE_GPU
    if (isEnabled[RUBRIC_GPU])
    {
        benchmark_GPUInit(M_U, (float*)M_U_GPU, workspaceElements);
        benchmark_GPU(matrixCount, N, (float*)M_U_GPU, (float*)Ain_GPU, (float*)Aout_GPU, Aref, Nmax, fidGPU,args,argCount);
    }
#endif

    //Benchmark MPI performance
#ifdef ENABLE_MPI
    if (isEnabled[RUBRIC_MPI])
    {
        benchmark_MPI(matrixCount, N, M_U, Ain, Aout, Aref, Nmax, fidMPI,args,argCount);
    }
#endif

    if (fidCPU)
    {
        fclose(fidCPU);
    }
    if (fidGPU)
    {
        fclose(fidGPU);
    }
    if (fidMPI)
    {
        fclose(fidMPI);
    }
    auto FinishInitialization = std::chrono::high_resolution_clock::now();

    _mm_free(M_0);
    _mm_free(M_U);

    _mm_free(work);
    _mm_free(Ain);
    _mm_free(Aout);
    _mm_free(Aref);

    _mm_free(argsIn);

#ifdef ENABLE_GPU
    if (isEnabled[RUBRIC_GPU])
    {
        cudaFree((void*)Aout_GPU);
        cudaFree((void*)Ain_GPU);
        cudaFree((void*)M_U_GPU);
    }
#endif

#ifdef ENABLE_MPI
    if (isEnabled[RUBRIC_MPI])
    {
        if (isMPIinited)
        {
            MPI_Finalize();
        }
    }
#endif

    if (isEnabled[RUBRIC_CPU] == 0 && isEnabled[RUBRIC_GPU] == 0 && isEnabled[RUBRIC_MPI])
    {
        std::cout << "All benchmarks were diabled. Nothing to benchmark.\n";
    }
}
