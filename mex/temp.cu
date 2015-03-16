#include "mex.h"
#include "gpu/mxGPUArray.h"

// helper
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

void mexFunction(int nout, mxArray *out[],
                 int nin,  mxArray const *in[])
{
  //// Prepare the Input
  mxGPUArray const *I;
  mxGPUArray const *p;
  mxGPUArray const *rcc;
  mxGPUArray const *rci;
  check_and_get_input(nin, in,  I,p,rcc,rci);
}