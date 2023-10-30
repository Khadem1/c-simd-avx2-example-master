#include <immintrin.h>
#include <stdio.h>

// gcc -mavx2 mask_load.c -o mask_load
int main() {

  int i;
  
  int int_array[8] = {100, 200, 300, 400, 500, 600, 700, 800};
  
__m256i mask = _mm256_setr_epi32(-20, -72, -48, -9, -100, 3, 5, 8);

__m256i result = _mm256_maskload_epi32(int_array, mask);
  
  /* Display the elements of the result vector */
  int* res = (int*)&result;
  printf("%d %d %d %d %d %d %d %d\n", res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);
  
  return 0;
}