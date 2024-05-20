#include <cuda_fp16.h>        // data types
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <cstdio>            // printf
#include <cstdlib>           // EXIT_FAILURE

#include <string.h>
#include <time.h>
#include <set>
#include <iostream>
#include <fstream>

#include "mmio.c"
#include "smsh.c"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::printf("CUDA API failed at line %d with error: %s (%d)\n",        \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        std::printf("CUSPARSE API failed at line %d with error: %s (%d)\n",    \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

const int EXIT_UNSUPPORTED = 2;

__half* createRandomArray(long int n) {
    __half* array = new __half[n];

    for (int i = 0; i < n; i++) { 
        array[i] = 1.0;
    }

    return array;
}

float* createRandomArray1(long int n) {
    float* array = new float[n];

    for (int i = 0; i < n; i++) { 
        array[i] = 1;
    }

    return array;
}

/* Finds the possible amount of column blocks the matrix can have */
int findMaxNnz(int *rowPtr, int *colIndex, int num_rows, int block_size) {

    int max = 0;
    int num_blocks = num_rows / block_size;

    std::set<int> mySet;

    for(int i=0; i < num_blocks; i++) {

        for (int j = 0; j<block_size; j++) {
            int id = (long int)block_size*i+j;
            
            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++)
                mySet.insert(colIndex[k]/block_size);
            
            if (mySet.size() > max)
                max = mySet.size();
        }
        mySet.clear();
    }

    return max*block_size;
}

/* Creates the array of block indexes for the blocked ell format */
int *createBlockIndex(int *rowPtr, int *colIndex, int num_rows, int block_size, int ell_cols) {

    long int mb = num_rows/block_size, nb = ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    int* hA_columns = new int[(long int)nb*mb]();
    long int ctr = 0;

    memset(hA_columns, -1, (long int) nb * mb * sizeof(int));
    std::set<int> mySet;

    /* Goes through the blocks of the matrix of block_size */
    for(int i=0; i<mb; i++) {

        /* Iterates through the rows of each block */
        for (int j = 0; j < block_size; j++) {
            long int id = (long int) block_size*i + j;
            int index = 0;
            if (id >= num_rows)
                break;

            /* Iterates over the nnzs of each row */
            for(int k=rowPtr[id]; k<rowPtr[id+1]; k++) {    
                index = (colIndex[k]/block_size);
                mySet.insert(index);
            }
        }
        for (std::set<int>::iterator it =mySet.begin(); it != mySet.end(); it++) {
	        int elem = *it;
	        hA_columns[ctr++] = elem;
        }
        
        ctr = (long int) i*nb+nb;
        mySet.clear();
    }
    return hA_columns; 
}

/* Creates the array of values for the blocked ell format */
__half *createValueIndex(int *rowPtr, int *colIndex, float *values, int *hA_columns, int num_rows, int block_size, int ell_cols) {

    /* Allocate enough memory for the array */
    __half* hA_values = new __half[(long int)num_rows * ell_cols]();

    long int mb = (long int) num_rows/block_size, nb = (long int) ell_cols/block_size;
    if (num_rows % block_size != 0)
        mb++;

    /* Set all values to 0 */
    memset(hA_values, 0, (long int) num_rows * ell_cols * sizeof(__half));

    /* Iterate the blocks in the y axis */
    for (int i=0; i<mb;i++){

        /* Iterate the lines of each block */
        for (int l = 0; l<block_size; l++) {
            int ctr = 0;

            /* Iterate the blocks in the block_id array (x axis) */
            for (int j = 0; j < nb; j++) {
                long int id = (long int) nb*i + j;
                if (hA_columns[id] == -1)
                    break;

                /* Iterate each line of the matrix */
                for(int k=rowPtr[(long int)i*block_size+l]; k<rowPtr[(long int)i*block_size+l+1]; k++) {  

                    /* If the element is not in the same block, skip*/
                    if (colIndex[k]/block_size > hA_columns[id])
                        break;
                    else if (colIndex[k]/block_size == hA_columns[id]) 
                        hA_values[(long int)i*ell_cols*block_size+l*ell_cols+j*block_size+(colIndex[k]-(hA_columns[id]*block_size))] = values[k];
                }
            }
        }
    }
    
    return hA_values;
}

int main(int argc, char *argv[]) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int A_num_rows, A_num_cols, nz, A_nnz;
    int i = 0, *I_complete, *J_complete;
    float *V_complete;
    
    	/* READ MTX FILE INTO CSR MATRIX */
    /************************************************************************************************************/
    if ((f = fopen(argv[1], "r")) == NULL)
    {
        printf("Could not locate the matrix file. Please make sure the pathname is valid.\n");
        exit(1);
    }

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(1);
    }
    matcode[4] = '\0';
    
    if ((ret_code = mm_read_mtx_crd_size(f, &A_num_rows, &A_num_cols, &nz)) != 0)
    {
        printf("Could not read matrix dimensions.\n");
        exit(1);
    }
    
    if ((strcmp(matcode, "MCRG") == 0) || (strcmp(matcode, "MCIG") == 0) || (strcmp(matcode, "MCPG") == 0) || (strcmp(matcode, "MCCG") == 0))
    {

        I_complete = (int *)calloc(nz, sizeof(int));
        J_complete = (int *)calloc(nz, sizeof(int));
        V_complete = (float *)calloc(nz, sizeof(float));

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }  
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            } 
            fscanf(f, "%*[^\n]\n");
            /* adjust from 1-based to 0-based */
            I_complete[i]--;
            J_complete[i]--;
        }
    }

    /* If the matrix is symmetric, we need to construct the other half */

    else if ((strcmp(matcode, "MCRS") == 0) || (strcmp(matcode, "MCIS") == 0) || (strcmp(matcode, "MCPS") == 0) || (strcmp(matcode, "MCCS") == 0) || (strcmp(matcode, "MCCH") == 0) || (strcmp(matcode, "MCRK") == 0) || (matcode[0] == 'M' && matcode[1] == 'C' && matcode[2] == 'P' && matcode[3] == 'S'))
    {

        I_complete = (int *)calloc(2 * nz, sizeof(int));
        J_complete = (int *)calloc(2 * nz, sizeof(int));
        V_complete = (float *)calloc(2 * nz, sizeof(float));

        int i_index = 0;

        for (i = 0; i < nz; i++)
        {
            if (matcode[2] == 'P') {
                fscanf(f, "%d %d", &I_complete[i], &J_complete[i]);
                V_complete[i] = 1;
            }
            else {
                fscanf(f, "%d %d %f", &I_complete[i], &J_complete[i], &V_complete[i]);
            }
                
            fscanf(f, "%*[^\n]\n");

            if (I_complete[i] == J_complete[i])
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
            }
            else
            {
                /* adjust from 1-based to 0-based */
                I_complete[i]--;
                J_complete[i]--;
                J_complete[nz + i_index] = I_complete[i];
                I_complete[nz + i_index] = J_complete[i];
                V_complete[nz + i_index] = V_complete[i];
                i_index++;
            }
        }
        nz += i_index;
    }
    else
    {
        printf("This matrix type is not supported: %s \n", matcode);
        exit(1);
    }

    /* sort COO array by the rows */
    if (!isSorted(J_complete, I_complete, nz)) {
        quicksort(J_complete, I_complete, V_complete, nz);
    }
    
    /* Convert from COO to CSR */
    int* rowPtr = new int[A_num_rows + 1]();
    int* colIndex = new int[nz]();
    float* values = new float[nz]();

    for (i = 0; i < nz; i++) {
        colIndex[i] = J_complete[i];
        values[i] = V_complete[i];
        rowPtr[I_complete[i] + 1]++;
    }
    for (i = 0; i < A_num_rows; i++) {
        rowPtr[i + 1] += rowPtr[i];
    }
    A_nnz = nz;

    free(I_complete);
    free(J_complete);
    free(V_complete);
    fclose(f);
    /* MTX READING IS FINISH */
    /************************************************************************************************************/

    // Host problem definition
    int   A_ell_blocksize = 32;
    
    //--------------------------------------------------------------------------
    // Pad matrix with extra rows and columns to be multiple of block size
    int * rowPtr_pad;
    int remainder = A_num_rows % A_ell_blocksize;
    if (remainder != 0) {
        A_num_rows = A_num_rows + (A_ell_blocksize - remainder);
        A_num_cols = A_num_cols + (A_ell_blocksize - remainder);
        rowPtr_pad = new int[A_num_rows + 1];
        for (int i=0; i<A_num_rows - (A_ell_blocksize - remainder); i++)
            rowPtr_pad[i] = rowPtr[i];
        for (int j=A_num_rows - (A_ell_blocksize - remainder); j<A_num_rows + 1; j++)
            rowPtr_pad[j] = nz;
        delete[] rowPtr;
    } else {
        rowPtr_pad = rowPtr;
    }   

    //--------------------------------------------------------------------------
    // Split matrix into 2 csr's
    int k = 16, split = 1;
    int blocks_per_part = (A_num_rows/A_ell_blocksize) / k;
    if ((A_num_rows/A_ell_blocksize) % k != 0)
        blocks_per_part += 1;
    long int A1_rows = blocks_per_part*split*A_ell_blocksize;
    long int A2_rows = A_num_rows - A1_rows;

    int ctr = split * blocks_per_part * A_ell_blocksize;
    int *rowPtr_part1 = new int[A1_rows + 1];
    
    for (int j=0; j<A1_rows; j++)
        rowPtr_part1[j] = rowPtr_pad[j];
    
    rowPtr_part1[A1_rows] = rowPtr_pad[A1_rows];
    A_nnz = rowPtr_part1[A1_rows];
    int* colIndex1 = new int[A_nnz]();
    float* values1 = new float[A_nnz]();

    for (int j=0; j<A_nnz; j++){
        colIndex1[j] = colIndex[j];
        values1[j] = values[j];
    }

    //--------------------------------------------------------------------------
    // Build blocked ell of the second csr
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = 32;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    long int   B_size          = (long int) ldb * B_num_cols;
    long int   C_size          = (long int) ldc * B_num_cols;

    /* Initialize variables */
    cusparseHandle_t     handle2;
    cusparseSpMatDescr_t matA2[k-split];
    cusparseDnMatDescr_t matB2, matC2[k-split];
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    int    *dA2_columns[k-split];
    __half *dA2_values[k-split], *dB2, *dC2[k-split];

    //--------------------------------------------------------------------------
    // Create dense vectors for spmm
    __half *hB2            = createRandomArray(B_size);
    float *hB1            = createRandomArray1(B_size);
    float *hC1            = new float[(long int) A1_rows*B_num_cols*sizeof(float)];

    float alpha           = 1.0f;
    float beta            = 0.0f;

    // Allocate and copy memory in GPU for dense vectors
    CHECK_CUDA( cudaMalloc((void**) &dB2, (long int) B_size * sizeof(__half)) )
    CHECK_CUDA( cudaMemcpy(dB2, hB2, (long int) B_size * sizeof(__half),
                        cudaMemcpyHostToDevice) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB2, A_num_cols, B_num_cols, ldb, dB2,
					CUDA_R_16F, CUSPARSE_ORDER_COL) )

    int total_blocks = 0;
    /* Create blocked ELL for each partition of matrix A */
    for (int i=0; i<k-split; i++){
        int *rowPtr_part = new int[blocks_per_part*A_ell_blocksize + 1];
        int A_rows = 0;
        for (int j=ctr; j<ctr+blocks_per_part*A_ell_blocksize; j++){
            if (j >= A_num_rows)
                break;
            rowPtr_part[j - ctr] = rowPtr_pad[j];
            A_rows++;
        }    
        ctr += A_rows;
        rowPtr_part[A_rows] = rowPtr_pad[ctr];
        long int nnzs_part = rowPtr_part[A_rows] - rowPtr_part[0];

        // Create blocked ELL vectors for partition
        int   A2_ell_cols      = findMaxNnz(rowPtr_part, colIndex, A_rows, A_ell_blocksize);
        double   A_num_blocks    = (double)A2_ell_cols * (double)A_rows /
                            (A_ell_blocksize * A_ell_blocksize);
        int   *hA2_columns     = createBlockIndex(rowPtr_part, colIndex, A_rows, A_ell_blocksize, A2_ell_cols);
        __half *hA2_values     = createValueIndex(rowPtr_part, colIndex, values, hA2_columns, A_rows, A_ell_blocksize, A2_ell_cols);

	    __half *hC2 	      = new __half[(long int) A_rows * B_num_cols * sizeof(__half)];

        // Allocate and copy memory in GPU for blocked ELL vectors
        CHECK_CUDA( cudaMalloc((void**) &dA2_columns[i], (long int) A_num_blocks * sizeof(int)) )
        CHECK_CUDA( cudaMalloc((void**) &dA2_values[i],
                                        (long int) A2_ell_cols * A_rows * sizeof(__half)) )
        CHECK_CUDA( cudaMalloc((void**) &dC2[i], (long int) A_rows * B_num_cols * sizeof(__half)) )
        
        CHECK_CUDA( cudaMemcpy(dA2_columns[i], hA2_columns,
                            (long int) A_num_blocks * sizeof(int),
                            cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dA2_values[i], hA2_values,
                            (long int) A2_ell_cols * A_rows * sizeof(__half),
                            cudaMemcpyHostToDevice) )
        CHECK_CUDA( cudaMemcpy(dC2[i], hC2, (long int) A_rows * B_num_cols * sizeof(__half),
                            cudaMemcpyHostToDevice) )
                            
        CHECK_CUSPARSE( cusparseCreateDnMat(&matC2[i], A_rows, B_num_cols, A_rows, dC2[i],
					CUDA_R_16F, CUSPARSE_ORDER_COL) )
        //--------------------------------------------------------------------------
        
        CHECK_CUSPARSE( cusparseCreate(&handle2) )
        // Create sparse matrix A in blocked ELL format
        CHECK_CUSPARSE( cusparseCreateBlockedEll(
                                        &matA2[i],
                                        A_rows, A_num_cols, A_ell_blocksize,
                                        A2_ell_cols, dA2_columns[i], dA2_values[i],
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) )
        
        // allocate an external buffer if needed
        CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                    handle2,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA2[i], matB2, &beta, matC2[i], CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
        CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

        //std::cout << A_num_blocks << std::endl;
        total_blocks += A_num_blocks;
        double density_part = (double) nnzs_part/A_num_blocks;
        std::cout << total_blocks << " density: " << density_part <<std::endl;
    }

    delete[] rowPtr_pad;
    delete[] colIndex;
    delete[] values;
    //--------------------------------------------------------------------------
    // Check compute capability
    cudaDeviceProp props;
    CHECK_CUDA( cudaGetDeviceProperties(&props, 0) )
    if (props.major < 7) {
      std::printf("cusparseSpMM with blocked ELL format is supported only "
                  "with compute capability at least 7.0\n");
      return EXIT_UNSUPPORTED;
    }
    //--------------------------------------------------------------------------
    // Device memory management CSR
    int   *hA1_csrOffsets = rowPtr_part1;
    int   *hA1_columns    = colIndex1;
    float *hA1_values     = values1;
    int   *dA1_csrOffsets, *dA1_columns;
    float *dA1_values, *dB1, *dC1;
    CHECK_CUDA( cudaMalloc((void**) &dA1_csrOffsets,
                           (long int)(A1_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA1_columns, (long int)A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA1_values,  (long int)A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB1, (long int) B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC1,         (long int)A1_rows*B_num_cols * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA1_csrOffsets, hA1_csrOffsets,
                           (long int)(A1_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA1_columns, hA1_columns, (long int)A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA1_values, hA1_values, (long int)A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB1, hB1, (long int) B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC1, hC1, (long int) A1_rows*B_num_cols * sizeof(float),
			   cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs CSR
    cusparseHandle_t     handle1 = NULL;
    cusparseSpMatDescr_t matA1;
    cusparseDnMatDescr_t matB1, matC1;
    void*                dBuffer1    = NULL;
    size_t               bufferSize1 = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle1) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA1, A1_rows, A_num_cols, A_nnz,
                                      dA1_csrOffsets, dA1_columns, dA1_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB1, A_num_cols, B_num_cols, ldb, dB1,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC1, A1_rows, B_num_cols, A1_rows, dC1,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle1,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA1, matB1, &beta, matC1, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize1) )
    CHECK_CUDA( cudaMalloc(&dBuffer1, bufferSize1) )
    //---------------------------------------------------------------------------

    struct timespec t_start, t_end;
    double elapsedTime, searchTime1, searchTime2;
    int numRuns=0;

    clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp
    // execute csr SpMM
    while (1) {
        CHECK_CUSPARSE( cusparseSpMM(handle1,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA1, matB1, &beta, matC1, CUDA_R_32F,
                                    CUSPARSE_SPMM_ALG_DEFAULT, dBuffer1) )
        cudaDeviceSynchronize();
        numRuns++;

        clock_gettime(CLOCK_MONOTONIC, &t_end);         // final timestamp
        elapsedTime = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
        if(elapsedTime > 5.0f) {        // changed from 1 sec to 5 sec
            break;
        }        
    }
    searchTime1 = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000))) / numRuns;

    // execute blocked ell SpMM
    for (int i=0; i<k-split; i++) {
        numRuns = 0;
        clock_gettime(CLOCK_MONOTONIC, &t_start);       // initial timestamp
        while (1) {
            CHECK_CUSPARSE( cusparseSpMM(handle2,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA2[i], matB2, &beta, matC2[i], CUDA_R_32F,
                                        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )
            cudaDeviceSynchronize();
            numRuns++;

            clock_gettime(CLOCK_MONOTONIC, &t_end);         // final timestamp
            elapsedTime = ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000)));
            if(elapsedTime > 5.0f) {        // changed from 1 sec to 5 sec
                break;
            }        
        }
        searchTime2 += ((t_end.tv_sec + ((double) t_end.tv_nsec / 1000000000)) - (t_start.tv_sec + ((double) t_start.tv_nsec / 1000000000))) / numRuns;
    }
    
    double tflops_bell = (2 * ((double)total_blocks) * ((double)A_ell_blocksize*A_ell_blocksize) * ((double)B_num_cols) / 1000000000000) / searchTime2;
    std::cout << argv[1];
    printf(" Time (seconds):\t%.6f\tperf (TFLOPS):\t%.6f\n", searchTime1 + searchTime2, tflops_bell);

    // destroy matrix/vector descriptors
    for (int i=0; i<k-split; i++) {
        CHECK_CUSPARSE( cusparseDestroySpMat(matA2[i]) )
        CHECK_CUSPARSE( cusparseDestroyDnMat(matC2[i]) )
    } 
    CHECK_CUSPARSE( cusparseDestroySpMat(matA1) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB2) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB1) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC1) )
    CHECK_CUSPARSE( cusparseDestroy(handle2) )
    CHECK_CUSPARSE( cusparseDestroy(handle1) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC1, dC1, (long int) A1_rows*B_num_cols * sizeof(float),
                           cudaMemcpyDeviceToHost) )

    /*std::ofstream outputFile("output.txt");
    for(int i=0; i<A1_rows*B_num_cols; i++)
    	outputFile << hC1[i] << std::endl;
    for(int i=0; i<A2_rows*B_num_cols; i++)
	outputFile << hC2[i] << std::endl; 
    outputFile.close();*/
    
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dBuffer1) )
    for (int i=0; i<k-split; i++) {
        CHECK_CUDA( cudaFree(dA2_columns[i]) )
        CHECK_CUDA( cudaFree(dA2_values[i]) )
        CHECK_CUDA( cudaFree(dC2[i]) )
    }
    CHECK_CUDA( cudaFree(dA1_columns) )
    CHECK_CUDA( cudaFree(dA1_values) )
    CHECK_CUDA( cudaFree(dB2) )
    CHECK_CUDA( cudaFree(dB1) )
    CHECK_CUDA( cudaFree(dC1) )
    
    delete[] hB1;
    delete[] hB2;
    delete[] hC1;
    return EXIT_SUCCESS;
}
