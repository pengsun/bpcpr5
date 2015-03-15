#include "mex.h"


void check_and_get_input (int nin, mxArray const *in[],
                          mxGPUArray &*I, 
                          mxGPUArray &*p,
                          mxGPUArray &*rcc,
                          mxArray    &*rci)
{

}

// [f,ind] = get_pixval(I, p, rcc, rci)
// f:   [MLN]     features
// ind: [MLN]     the linear index
// I:   [H,W,3,N] image array
// p:   [2,L,N]   points
// rcc: [K, ML]   combination coefficients
// rci: [K, ML]   non zero elements index
void mexFunction(int nout, mxArray *out[],
                 int nin,  mxArray const *in[])
{
  //// Prepare the Input
  mxGPUArray *I, *p, *rcc;
  mxArray    *rci;
  check_and_get_input(nin, in,  I,p,rcc,rci);

  //// Create the Output
  int ML = 10; // TODO
  int N  = 10; // TODO
  mxArray *ind = 0; // TODO
  mxGPUArray *f = 0; // TODO

  // do the job

}