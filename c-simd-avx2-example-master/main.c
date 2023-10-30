#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>

/**
 * Register (xmm) 128-bit (16 bytes), Register 256-bit (32bytes) (ymm), Regsiter 512-bits (64 bytes) (zmm)
 * xmm can have 2 doubles or 4 floats or 4 integers 
 * ymm can have 4 doubles or 8 floats or 8 integers
 * zmm can have 8 double or 16 floats or 16 integers 
*/
//gcc -mavx2 main.c -o output
#define ARRAY_LENGTH 8

int main(int argc, char* argv[]) {

    __m256i first = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 70, 80);
    __m256i second = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 5);
    __m256i result = _mm256_add_epi32(first, second);

    int* values = (int*) &result;

    for ( unsigned short i = 0; i < ARRAY_LENGTH; i += 1) {
        printf("%d \n", values[i]);
    }
    double vector [4] = {1.000001,1.000001,1.000001,1.000001}; 
    //float * ps = (float*)_mm_malloc(40, 32);
    double * pd = (double*)_mm_malloc(40, 32);
    for (int i =0; i<=3;i++){
        *(pd+i) = 1.000001; 
    }
    __m256d dbl_vector = _mm256_setzero_pd();
            dbl_vector = _mm256_load_pd (pd);       // load from an aligned memory
    __m256d dbl_vector4 = _mm256_loadu_pd (vector);   // load from an unaligned memory 
            //_mm256_storeu_pd(pd, dbl_vector);
    __m256d dbl_vector2 = _mm256_set_pd (1,2,3,4); 
    __m256d dbl_vector3 = _mm256_add_pd (dbl_vector,dbl_vector2); 
    __m256d dbl_vector5 = _mm256_add_pd (dbl_vector4, dbl_vector2);
    double *dbl = (double*) &dbl_vector3; 
    double *dbl2 = (double *) &dbl_vector5; 
    printf ("Double Addition with SIMD programming \n"); 
    for (int i=0; i<=3; i++){
        printf(" %f ",dbl[i]); 
       // printf(" %f ", dbl2[i]);
    }  
    printf ("\n"); 

    for (int i=0; i<=3; i++){
    //    printf(" %f ",dbl[i]); 
        printf(" %f ", dbl2[i]);
    } 

    printf ("\n"); 

    /**
     *  Loading pd from memory with mask;  
     *  The numbers if are in negative with unmask 
     *  the corresponding number ( mask bits associated with each data element.). 
     *  2 mask elementes are used with one vector element to masked it. 
     * 
    */
    double vector_2 [4] = {1.000001,2.000001,3.000001,4.000001}; 
    __m256i mask = _mm256_set_epi32(-20, -72, -5, -8,-1,-2,-3,-4);  
    __m256d dblm_vector = _mm256_maskload_pd(vector_2,mask); 
    double *dblm = (double *) &dblm_vector; 
    for (int i=0; i<=3; i++){
    //    printf(" %f ",dbl[i]); 
        printf(" %f ", dblm[i]);
    } 

    printf ("\n"); 

    /** 
     * Add two vectors containing shorts with saturation, 
     * The __mm256_adds_epi8() adds two short and return the results, if the results is greater than 
     * 127, it will be reduced to the maximum value of the short int, i.e., to 127. If each individual 
     * element is greater than 127 and you have add a value, then it will not be reduced to 127.    
    */

    __m256i dblm_vector_v1 = _mm256_set_epi32(10, 20, 30, 40, 50, 60, 130, 126);
    __m256i dblm_vector_v2 = _mm256_set_epi32(5, 5, 5, 5, 5, 5, 5, 50);
    __m256i dblm_vector_v3 = _mm256_adds_epi8(dblm_vector_v1, dblm_vector_v2);

    printf("The following represents the addition of two vectors with saturation \n"); 
    int *s1 = (int *) &dblm_vector_v3;

    for (int s=0; s<=7; s++){
        printf("%d ",s1[s]); 
    }    
    
    printf("\n Without staturation \n"); 

    __m256i dblm_vector_pf = _mm256_add_epi8(dblm_vector_v1,dblm_vector_v2);
    int *vector_pf = (int *) & dblm_vector_pf;  

    for (int w = 0; w <= ARRAY_LENGTH; w++)
    {
        printf ("%d ", vector_pf[w]);   
    }
    printf("\n"); 
    
    /**
     * __mm256_hadd_epi16 () add two integer vectors horizontally
     * 
    */
    printf("Adding vectors with __mm356_hadd_epi26() \n"); 
    __m256i dblm_v1 = _mm256_set_epi32(1,2,3,4,5,6,7,8); 
    __m256i dblm_v2 = _mm256_set_epi32(1,2,3,4,5,6,7,8);
    __m256i dblm_v3 = _mm256_hadd_epi32(dblm_v1,dblm_v2);
    __m256i dblmm = _mm256_hsub_epi32(dblm_v1, dblm_v2); 

    int *dblm_v4 = (int *) &dblm_v3, *dblm_v5 = (int *) &dblmm; 

    for (int j = 0; j <= ARRAY_LENGTH; j++)
    {
        printf ("%d ", dblm_v4[j]); 
    }
    printf("\n Subtracting vectors with __m256_hsub_epi16 \n");

    for (int s = 0; s <= ARRAY_LENGTH; s++)
    {
        printf("%d ", dblm_v5[s]);
    }

    /**
     * The following will represent the function _mm356_addsub_pd()
     * Add and subtract two floating-point vectors 
     * */     
    __m256d dblm_vec1 = _mm256_set_pd(0.1,0.2,0.3,0.4);
    __m256d dblm_vec2 = _mm256_set_pd(0.5,0.6,0.7,0.8);
    __m256d dblm_vec3 = _mm256_add_pd(dblm_vec1,dblm_vec2);

    double *dblm_vc3 = (double *) &dblm_vec3;  
    printf("\n Using _mm256_addsub_pd() \n"); 

    for (int i = 0; i <= ARRAY_LENGTH; i++)
    {
        printf("%f ", dblm_vc3[i]); 
    }

    
    unsigned char eth_hdr[16] = {
    0x28, 0xa6, 0xdb, 0x40, 0x6f, 0x97, 0x10, 0x6f, 
    0xd9, 0x0c, 0x05, 0x77, 0x08, 0x00};  

    printf("\n macswap with SIMD \n"); 
    printf("Original eth header \n"); 

    for (int i = 0; i < 15; i++)
    {
        printf("%x ",eth_hdr[i]); 
    }
    __m128i addr0, addr1, addr2, addr3;
    /**
	 * shuffle mask be used to shuffle the 16 bytes.
	 * byte 0-5 wills be swapped with byte 6-11.
	 * byte 12-15 will keep unchanged.
	 */
	__m128i shfl_msk = _mm_set_epi8(15, 14, 13, 12,
					5, 4, 3, 2,
					1, 0, 11, 10,
					9, 8, 7, 6);
    
    addr0 = _mm_loadu_si128((__m128i *)eth_hdr);
    addr0 = _mm_shuffle_epi8(addr0, shfl_msk);
    _mm_storeu_si128((__m128i *)eth_hdr, addr0);
    
    char *eth_shuffle = (char *) &addr0;

    printf("\n Eth header After macswap with shuffle \n"); 

    for (int i = 0; i < 15; i++)
    {
        printf("%x ",eth_hdr[i]); 
    }

    printf("\n Permute Intrinsics"); 

    /* flags for desc: 0, 2, 4, 6, 1, 3, 5, 7 */
		__m256i flags0_7 = _mm256_set_epi32(0, 2, 4, 6,1, 3, 5, 7); 
		/*
		 * Swap to reorder flags in this order: 1, 3, 5, 7, 0, 2, 4, 6
		 * This order simplifies blend operations way below that
		 * produce 'rearm' data for each mbuf.
		 */
		flags0_7 = _mm256_permute4x64_epi64(flags0_7,
			(1 << 6) + (0 << 4) + (3 << 2) + 2);
        
        int *permute_v = (int *) &flags0_7;

        for (int j = 0; j <= 7; j++)
            {
                printf ("%d ", permute_v[j]); 
            }
    printf("\n"); 

        //_mm_free(ps);
    _mm_free(pd);

    return 0;
}
