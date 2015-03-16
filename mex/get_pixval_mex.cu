#include "mex.h"
#include "gpu/mxGPUArray.h"

// helper
void check_and_get_input (int nin, mxArray const *in[],
                          mxGPUArray *&I, 
                          mxGPUArray *&p,
                          mxGPUArray *&rcc,
                          mxArray    *&rci)
{
  if (nin != 4)
    mexErrMsgTxt("Incorrect arguments. [f,ind] = get_pixval(I, p, rcc, rci)");

  I   = in[0];
  p   = in[1];
  rcc = in[2];
  rci = in[3];

  if ( mxIsGPUArray(I) == 0 )
    mexErrMsgTxt("I must be a gpuArray.");
  if ( mxIsGPUArray(p) == 0 )
    mexErrMsgTxt("p must be a gpuArray.");
  if ( mxIsGPUArray(rcc) == 0 )
    mexErrMsgTxt("rcc must be a gpuArray.");
  if ( mxIsGPUArray(rci) == 1 )
    mexErrMsgTxt("rci must be a matlab Array.");

}

// get all the points pp by random combination
void get_pp_by_rc (mxGPUArray *p, mXGPUArray *rcc, mxArray *rci,
                   mxGPUArray *&pp)
{
  
}

// get the values and the index 
void get_ind_val (mxGPUArray *I, mxGPUArray *pp,
                  mxGPUArray *&f, mxGPUArray *&ind)
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
  size_t ML         = mxGetN(rcc);
  const mwSize *dim = mxGetDimensions(p);
  mwSize N          = *(dim + 2); 
  mxGPUArray *f = mxGPUCreateGPUArray (1, &(ML*N), mxSINGLE_CLASS, mxREAL, // [MLN]
                                       MX_GPU_DO_NOT_INITIALIZE); 
  mxArray *ind  = mxCreateNumericArray(1, &(ML*N), mxINT32_CLASS, mxREAL); // [MLN]


  //// do the job

  // get all the points pp: [2, ML, N]
  mwSize pp_dim[3];
  pp_dim[0] = 2;
  pp_dim[1] = ML;
  pp_dim[2] = N;
  mxGPUArray *pp = mxGPUCreateGPUArray (3, pp_dim, mxSINGLE_CLASS, mxREAL, // [MLN]
                                        MX_GPU_DO_NOT_INITIALIZE);
  get_pp_by_rc (p,rcc,rci,  pp);

  // get the linear index and the values
  get_ind_val (I,pp, f,ind);

  return;
}
