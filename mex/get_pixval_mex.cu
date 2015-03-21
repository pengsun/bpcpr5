#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "tmwtypes.h"

// Block Size
static const int BS = 16;

__global__ void cuda_get_pp_by_rc (const int      K,
                                   const int      L,
                                   const int      ML,
                                   const int      N,
                                   const float    *p,
                                   const float    *rcc,
                                   const uint32_T *rci,
                                   float          *pp)
{
  //// make sure we're working within the range of pp

  // the index along dim1 of pp: 0 or 1
  int dim_pnt = threadIdx.z;

  // the index along dim2 of pp: ml
  int sub_ml    = threadIdx.x;
  int blkcnt_ml = blockIdx.x;
  int ml        = BS*blkcnt_ml + sub_ml;
  if (ml >= ML) return;

  // the index along dim3 of pp: n
  int sub_n    = threadIdx.y;
  int blkcnt_n = blockIdx.y;
  int n        = BS*blkcnt_n + sub_n;
  if (n >= N) return;

  
  //// do the job

  // initialize the pp value
  float val = 0.0;

  // weighted sum of the K points
  for (int k = 0; k < K; ++k) {
    // which point
    int i_pnt = *(rci + k + ml*K);
    i_pnt -= 1; // matlab 1-base -> C 0-base

    // the point value
    float pv = *(p + dim_pnt + i_pnt*2 + n*2*L);

    // what coefficient
    float w = *(rcc + k + ml*K);

    // the output point value
    val += pv*w;
  } // for k

  // write to the target
  *(pp + dim_pnt + ml*2 + n*2*ML) = val;

  __syncthreads();

}

__global__ void cuda_get_ind_val (const int   H,
                                  const int   W,
                                  const int   ML,
                                  const int   N,
                                  const float *I,
                                  const float *pp,
                                  float       *f,
                                  uint32_T    *ind )
{
  //// make sure we're working within the range of pp

  // the index along dim2 of pp: ml
  int sub_ml    = threadIdx.x;
  int blkcnt_ml = blockIdx.x;
  int ml        = BS*blkcnt_ml + sub_ml;
  if (ml >= ML) return;

  // the index along dim3 of pp: n
  int sub_n    = threadIdx.y;
  int blkcnt_n = blockIdx.y;
  int n        = BS*blkcnt_n + sub_n;
  if (n >= N) return;


  //// do the job

  // get (py, px) from pp [2, ML, N],
  int i_pntx = 0 + ml*2 + n*2*ML; // the convention of the pp storage: x first, y second
  int i_pnty = 1 + ml*2 + n*2*ML;
  // normalized coordinate -> integer coordinate
  int py = int( float(H) * pp[i_pnty] ); // [0,1] -> {0,1,...,H-1}
  int px = int( float(W) * pp[i_pntx] ); // [0,1] -> {0,1,...,W-1}
  py = (py<H) ? (py) : (H-1); // make it in the range
  px = (px<W) ? (px) : (W-1); 
  // convert it to linear index to the image I,
  // i.e., the linear index for (py,px,0,n) at I [H,W,3,N] 
  int i_pixval = py + px*H + n*H*W*3; // py + px*H + 0*W*H + n*H*W*3;


  //// fill the output
  int i_out = ml + ML*n; // the linear index for the output 
  f[i_out] = I[i_pixval];  // fill the feature f
  ind[i_out] = (uint32_T)(i_pixval + 1);  // fill the index ind: 0-base -> 1-base

  __syncthreads();

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
  //// raw pointer
  const float    *ptr_p   = (const float*)    ( mxGPUGetDataReadOnly(p) );
  const float    *ptr_rcc = (const float*)    ( mxGPUGetDataReadOnly(rcc) );
  const uint32_T *ptr_rci = (const uint32_T*) ( mxGPUGetDataReadOnly(rci) );
  float          *ptr_pp  = (float*)          ( mxGPUGetData(pp) );

  //// auxiliary 
  const int K  = *( 0 + mxGPUGetDimensions(rcc) ); // rcc [K, ML]
  const int ML = *( 1 + mxGPUGetDimensions(rcc) );
  const int L  = *( 1 + mxGPUGetDimensions(p)   ); // p [2,L,N]
  const int N  = *( 2 + mxGPUGetDimensions(p)   ); // p [2,L,N]

  //// block and thread partition
  dim3 num_thd( BS, BS, 2 );
  dim3 num_blk( (ML+BS-1)/BS, (N+BS-1)/BS );

#ifndef NDEBUG
  mexPrintf("In get_pp_by_rc\n");
  mexPrintf("K = %d\n", K);
  mexPrintf("ML = %d\n", ML);
  mexPrintf("L = %d\n", L);
  mexPrintf("N = %d\n", N);
#endif // !NDEBUG

  cuda_get_pp_by_rc<<<num_blk, num_thd>>>(K,L,ML,N, ptr_p,ptr_rcc,ptr_rci,  ptr_pp);
}

// [f,ind] = get_ind_val(I,pp); Get the values and the index 
void get_ind_val (mxGPUArray const *I, 
                  mxGPUArray const *pp,
                  mxGPUArray       *f, 
                  mxGPUArray       *ind)
{
  // thread 
  dim3 num_thd(BS,BS);
  // block
  const int ML = *( 1 + mxGPUGetDimensions(pp) ); // p [2,L,N]
  const int N  = *( 2 + mxGPUGetDimensions(pp) ); 
  dim3 num_blk( (ML+BS-1)/BS, (N+BS-1)/BS );
  // image size
  const int H = *( 0 + mxGPUGetDimensions(I) ); // I [H, W, 3, N]
  const int W = *( 1 + mxGPUGetDimensions(I) ); 
  // raw pointer
  const float    *ptr_I   = (const float*) ( mxGPUGetDataReadOnly(I) );
  const float    *ptr_pp  = (const float*) ( mxGPUGetDataReadOnly(pp) );
  float          *ptr_f   = (float*)       ( mxGPUGetData(f) );
  uint32_T       *ptr_ind = (uint32_T*)    ( mxGPUGetData(ind) );

#ifndef NDEBUG
  mexPrintf("In get_ind_val\n");
  mexPrintf("H = %d\n", H);
  mexPrintf("W = %d\n", W);
  mexPrintf("ML = %d\n", ML);
  mexPrintf("N = %d\n", N);
#endif // !NDEBUG

  cuda_get_ind_val<<<num_blk, num_thd>>>(H,W,ML,N, ptr_I,ptr_pp,   ptr_f,ptr_ind);
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
