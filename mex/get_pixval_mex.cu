#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "tmwtypes.h"


void __global__ cuda_get_pp_by_rc (const int      K,
                                   const int      L,
                                   const float    *p,
                                   const float    *rcc,
                                   const uint32_T *rci,
                                   float          *pp)
{
  int ml = blockIdx.x;
  int ML = gridDim.x; // ???
  int i_pntdim = threadIdx.x;
  int n  = threadIdx.y;

  // initialize the pp value
  *(pp + i_pntdim + ml*2 + n*2*ML) = (float) 0.0;
  
  // accumulate the combination of the K points
  for (int k = 0; k < K; ++k) {
    *(pp + k) = (float) ( (k+1)*1.57 );

    // which point
    int i_pnt = *(rci + k + ml*K);
    i_pnt -= 1; // matlab 1-base -> C 0-base

    // the point value
    float pv = *(p + i_pntdim + i_pnt*2 + n*2*L);

    // what coefficient
    float w = *(rcc + k + ml*K);

    // the output point value
    *(pp + i_pntdim + ml*2 + n*2*ML) += pv*w;
    
  }

}

void __global__ cuda_get_ind_val (const float    *I,
                                  const float    *pp,
                                  const float    *f,
                                  const uint32_T *ind )
{

}


// [I,p,rcc,rci] = check_and_get_input(nin,in); Helper
void check_and_get_input (int              nin, 
                          mxArray    const *in[],
                          mxGPUArray const *&I, 
                          mxGPUArray const *&p,
                          mxGPUArray const *&rcc,
                          mxGPUArray const *&rci)
{
  if (nin != 4)
    mexErrMsgTxt("Incorrect arguments. [f,ind] = get_pixval(I, p, rcc, rci)");

  //// check if gpuArray
  if ( mxIsGPUArray( in[0] ) == 0 ) mexErrMsgTxt("I must be a gpuArray.");
  if ( mxIsGPUArray( in[1] ) == 0 ) mexErrMsgTxt("p must be a gpuArray.");
  if ( mxIsGPUArray( in[2] ) == 0 ) mexErrMsgTxt("rcc must be a gpuArray.");
  if ( mxIsGPUArray( in[3] ) == 0 ) mexErrMsgTxt("rci must be a gpuArray."); 
  
  //// fetch the results
  I   = mxGPUCreateFromMxArray( in[0] );
  p   = mxGPUCreateFromMxArray( in[1] );
  rcc = mxGPUCreateFromMxArray( in[2] );
  rci = mxGPUCreateFromMxArray( in[3] );

  //// check the types
  if (mxGPUGetClassID(I)   != mxSINGLE_CLASS ) mexErrMsgTxt("I must be the type single.");
  if (mxGPUGetClassID(p)   != mxSINGLE_CLASS ) mexErrMsgTxt("p must be the type single.");
  if (mxGPUGetClassID(rcc) != mxSINGLE_CLASS ) mexErrMsgTxt("rcc must be the type single.");
  if (mxGPUGetClassID(rci) != mxUINT32_CLASS ) mexErrMsgTxt("rci must be the type uint32.");
}

// pp = get_pp_by_rc(p,rcc,rci); Get all the points pp by random combination
void get_pp_by_rc (mxGPUArray const *p, 
                   mxGPUArray const *rcc, 
                   mxGPUArray const *rci,
                   mxGPUArray       *pp)
{
  const float    *ptr_p   = (const float*)    ( mxGPUGetDataReadOnly(p) );
  const float    *ptr_rcc = (const float*)    ( mxGPUGetDataReadOnly(rcc) );
  const uint32_T *ptr_rci = (const uint32_T*) ( mxGPUGetDataReadOnly(rci) );
  float          *ptr_pp  = (float*)          ( mxGPUGetData(pp) );

  const int K  = *( 0 + mxGPUGetDimensions(rcc) ); // rcc [K, ML]
  const int ML = *( 1 + mxGPUGetDimensions(rcc) );
  const int L  = *( 1 + mxGPUGetDimensions(p)   ); // p [2,L,N]
  const int N  = *( 2 + mxGPUGetDimensions(p)   ); // p [2,L,N]
  dim3 num_thd(2, N); // 2 dimensional point

#ifndef NDEBUG
  mexPrintf("In get_pp_by_rc\n");
  mexPrintf("K = %d\n", K);
  mexPrintf("ML = %d\n", ML);
  mexPrintf("L = %d\n", L);
  mexPrintf("N = %d\n", N);
#endif // !NDEBUG

  cuda_get_pp_by_rc<<<ML, num_thd>>>(K,L,ptr_p,ptr_rcc,ptr_rci,  ptr_pp);
}

// [f,ind] = get_ind_val(I,pp); Get the values and the index 
void get_ind_val (mxGPUArray const *I, 
                  mxGPUArray const *pp,
                  mxGPUArray const *f, 
                  mxGPUArray const *ind)
{
  const float    *ptr_I   = (const float*)    ( mxGPUGetDataReadOnly(I) );
  const float    *ptr_pp  = (const float*)    ( mxGPUGetDataReadOnly(pp) );
  const float    *ptr_f   = (const float*)    ( mxGPUGetDataReadOnly(f) );
  const uint32_T *ptr_ind = (const uint32_T*) ( mxGPUGetDataReadOnly(ind) );

  cuda_get_ind_val<<<1, 1>>>(ptr_I,ptr_pp,  ptr_f,ptr_ind);
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
  mxGPUArray const *I;
  mxGPUArray const *p;
  mxGPUArray const *rcc;
  mxGPUArray const *rci;
  check_and_get_input(nin, in,  I,p,rcc,rci);


  //// Create the Output
  const mwSize *ddd = mxGPUGetDimensions(rcc);
  size_t ML         = *(ddd + 1);
  const mwSize *dim = mxGPUGetDimensions(p);
  mwSize N          = *(dim + 2); 
  mwSize dimo[1];
  dimo[0] = ML*N;
  mxGPUArray *f   = mxGPUCreateGPUArray(1, dimo, mxSINGLE_CLASS, mxREAL, // [MLN]
                                        MX_GPU_DO_NOT_INITIALIZE); 
  out[0]          = mxGPUCreateMxArrayOnGPU(f);
  mxGPUArray *ind = mxGPUCreateGPUArray(1, dimo, mxUINT32_CLASS, mxREAL, // [MLN]
                                        MX_GPU_DO_NOT_INITIALIZE); 
  out[1]          = mxGPUCreateMxArrayOnGPU(ind);


  //// do the job

  // get all the points pp: [2, ML, N]
  mwSize pp_dim[3];
  pp_dim[0] = 2;
  pp_dim[1] = ML;
  pp_dim[2] = N;
  mxGPUArray *pp = mxGPUCreateGPUArray (3, pp_dim, mxSINGLE_CLASS, mxREAL, // [MLN]
                                        MX_GPU_DO_NOT_INITIALIZE);
  get_pp_by_rc (p,rcc,rci,  pp);
  out[2] = mxGPUCreateMxArrayOnGPU(pp);

  // get the linear index and the values
  get_ind_val (I,pp, f,ind);

  // cleanup !!!
  mxGPUDestroyGPUArray(I);
  mxGPUDestroyGPUArray(p);
  mxGPUDestroyGPUArray(rcc);
  mxGPUDestroyGPUArray(rci);
  mxGPUDestroyGPUArray(pp);
  mxGPUDestroyGPUArray(f);
  mxGPUDestroyGPUArray(ind);

  return;
}
