// Doing the operation:
//
//                         | a a a a |  | y |
// x * A * y = [ x x x x ] | a a a a |  | y |
//                         | a a a a |  | y |
//                         | a a a a |  | y |
//
// with SIMD intrinics (specifically AVX).
//
// adapted from https://gist.github.com/rygorous/4172889
//
// compile with cc -march=native -O3 simd-vmv.c -o simd-vmv
//
// On my haswell Core i7-4500U CPU @ 1.80GHz, The reference implementation
// is actually slightly faster or about the same. Results:
//
// all ok.
//         ref: 10.46 cycles
//         avx: 11.34 cycles

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// reference implementation
void vecmatvec_ref(const double* restrict x, const double* restrict A,
                   const double* restrict y, double* restrict out)
{
    double sum = 0.0;
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            sum += x[i] * A[4*i+j] * y[j];
    *out = sum;
}

// maybe faster with aligned mem?
void vecmatvec_avx(const double* restrict x, const double* restrict A,
                   const double* restrict y, double* restrict out)
{
    asm volatile ("# avx code begin");  // looking at assembly with gcc -S
    __m256d yrow = _mm256_loadu_pd(y);
    __m256d xrow = _mm256_loadu_pd(x);
    
    __m256d a = _mm256_mul_pd(_mm256_loadu_pd(A), yrow);
    __m256d b = _mm256_mul_pd(_mm256_loadu_pd(A+4), yrow);
    __m256d c = _mm256_mul_pd(_mm256_loadu_pd(A+8), yrow);
    __m256d d = _mm256_mul_pd(_mm256_loadu_pd(A+12), yrow);

    // our task now is to get {sum(a), sum(b), sum(c), sum(d)}
    // This is tricky because there is no hadd instruction for avx
    
    // {a[0]+a[1], b[0]+b[1], a[2]+a[3], b[2]+b[3]}
    __m256d sumab = _mm256_hadd_pd(a, b);   

    // {c[0]+c[1], d[0]+d[1], c[2]+c[3], d[2]+d[3]}
    __m256d sumcd = _mm256_hadd_pd(c, d);   

    // {a[0]+a[1], b[0]+b[1], c[2]+c[3], d[2]+d[3]}
    __m256d blend = _mm256_blend_pd(sumab, sumcd, 0b1100);

    // {a[2]+a[3], b[2]+b[3], c[0]+c[1], d[0]+d[1]}
    __m256d perm = _mm256_permute2f128_pd(sumab, sumcd, 0x21);

    // {sum(a), sum(b), sum(c), sum(d)}
    __m256d Ay = _mm256_add_pd(perm, blend);

    // Now we want to take the dotproduct: dot(x, Ay)

    __m256d e = _mm256_mul_pd(Ay, xrow);

    // horizontal sum of e: tricky again.
    // {e[0]+e[1], e[0]+e[1], e[2]+e[3], e[2]+e[3]}
    __m256d tmp = _mm256_hadd_pd(e, e);

    // {e[2]+e[3], e[2]+e[3]}
    __m128d e23 = _mm256_extractf128_pd(tmp, 1);
    __m128d result = _mm_add_pd(_mm256_castpd256_pd128(tmp), e23);

    _mm_storel_pd(out, result);
    asm volatile ("# avx code end");
}

//-----------------------------------------------------------------------------
// testing stuff

// x86_64 only: "read timestamp counter"
// from http://stackoverflow.com/questions/9887839/clock-cycle-count-wth-gcc
static __inline__ unsigned long long rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


static double frand()
{
    return (double)rand() / (double)RAND_MAX;
}

static void randmat(double *A)
{
    for (int i=0; i < 4; i++)
        for (int j=0; j < 4; j++)
            A[4*i+j] = frand();
}

static void randvec(double *x)
{
    for (int i=0; i < 4; i++)
        x[i] = frand();
}

int the_mask = 0; // global so the compiler can't be sure what its value is for opt.

static double run_ref(double *x, double *A, double *y, int count)
{
    double result = 0.0;
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        vecmatvec_ref(x+j, A+j, y+j, &result);
    }
    return result;
}

static double run_avx(double *x, double *A, double *y, int count)
{
    double result = 0.0;
    for (int i=0; i < count; i++)
    {
        int j = i & the_mask;
        vecmatvec_avx(x+j, A+j, y+j, &result);
    }
    return result;
}

int main(int argc, char **argv)
{
    static const struct {
        const char *name;
        void (*vecmatvec)(const double *x, const double *A, const double *y,
                          double *out)
    } variants[] = {
        { "ref",      vecmatvec_ref },
        { "avx",      vecmatvec_avx }
    };
    static const int nvars = (int) (sizeof(variants) / sizeof(*variants));
    
    srand(1);

    // correctness tests
    for (int i=0; i < 1000; i++)
    {
        double A[16];
        double x[4], y[4];
        randmat(A);
        randvec(x);
        randvec(y);
        double ref_out;
        vecmatvec_ref(x, A, y, &ref_out);
        double out;

        for (int j=0; j < nvars; j++)
        {
            variants[j].vecmatvec(x, A, y, &out);

            // not perfectly equal because we reordered floating point ops
            if (fabs(out - ref_out) > 1e-14) {
                fprintf(stderr, "%s fails test\n", variants[j].name);
                exit(1);
            }
        }
    }

    printf("all ok.\n");

    // perf tests
    // as usual with such microbenchmarks, this isn't measuring anything
    // terribly useful, but here goes.
    static const struct {
        const char *name;
        double (*run)(double *x, double *A, double *y, int count);
    } perf_variants[] = {
        { "ref",      run_ref },
        { "avx",      run_avx },
    };
    static const int nperfvars = (int) (sizeof(perf_variants) / sizeof(*perf_variants));
    
    /* 
    */
    
    double Ap[16];
    double xp[4], yp[4];
    double out;
    randmat(Ap);
    randvec(xp);
    randvec(yp);

    
    for (int i=0; i < nvars; i++)
    {
        static const int nruns = 4096;
        static const int muls_per_run = 4096;
        unsigned long long best_time = ~0ull;

        for (int run=0; run < nruns; run++)
        {
            unsigned long long time = rdtsc();
            perf_variants[i].run(xp, Ap, yp, muls_per_run);
            time = rdtsc() - time;
            if (time < best_time)
                best_time = time;
        }

        double cycles_per_run = (double) best_time / (double) muls_per_run;
        printf("%12s: %.2f cycles\n", perf_variants[i].name, cycles_per_run);
    }

    return 0;
}