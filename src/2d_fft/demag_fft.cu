/*
 * demag_test_v2.cu
 *
 * Standalone test for the efficient demag (v2):
 *   f̂ computed once, stays on device.
 *   Apply: all GPU — pack → FFT → multiply → IFFT → scatter.
 *   Zero D2H/H2D in the hot path.
 *
 * Same magnetization pattern and output format as demag_test.cu (v1),
 * so you can diff/compare in MATLAB:
 *
 *   nvcc -O3 -arch=sm_89 demag_test_v2.cu -lcufft -o demag_test_v2
 *   ./demag_test_v2 > out_v2.txt
 *   ./demag_test    > out_v1.txt
 *
 *   In MATLAB:
 *   A = load('out_v1.txt');   % skip header line
 *   B = load('out_v2.txt');
 *   max(max(abs(A(2:end,:) - B(2:end,:))))   % should be ~1e-10
 */

#include <cstdio>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

static const int    NN    = 64;
static const double THICK = 100.0;

/* ── ctt / calt: exact copy from professor's pseudocode (1-indexed dm[]) ── */
static void ctt(double b, double a, double sx, double sy, double dm[])
{
    double sz=0.0;
    double xn=sx-a,xp=sx+a,yn=sy-a,yp=sy+a,zn=sz-b,zp=sz+b;
    double xn2=xn*xn,xp2=xp*xp,yn2=yn*yn,yp2=yp*yp,zn2=zn*zn,zp2=zp*zp;
    double dnnn=std::sqrt(xn2+yn2+zn2),dpnn=std::sqrt(xp2+yn2+zn2);
    double dnpn=std::sqrt(xn2+yp2+zn2),dnnp=std::sqrt(xn2+yn2+zp2);
    double dppn=std::sqrt(xp2+yp2+zn2),dnpp=std::sqrt(xn2+yp2+zp2);
    double dpnp=std::sqrt(xp2+yn2+zp2),dppp=std::sqrt(xp2+yp2+zp2);
    dm[1]=std::atan(zn*yn/(xn*dnnn))-std::atan(zp*yn/(xn*dnnp))
         -std::atan(zn*yp/(xn*dnpn))+std::atan(zp*yp/(xn*dnpp))
         -std::atan(zn*yn/(xp*dpnn))+std::atan(zp*yn/(xp*dpnp))
         +std::atan(zn*yp/(xp*dppn))-std::atan(zp*yp/(xp*dppp));
    dm[2]=std::log((dnnn-zn)/(dnnp-zp))-std::log((dpnn-zn)/(dpnp-zp))
         -std::log((dnpn-zn)/(dnpp-zp))+std::log((dppn-zn)/(dppp-zp));
    dm[3]=std::log((dnnn-yn)/(dnpn-yp))-std::log((dpnn-yn)/(dppn-yp))
         -std::log((dnnp-yn)/(dnpp-yp))+std::log((dpnp-yn)/(dppp-yp));
    dm[4]=std::log((dnnn-zn)/(dnnp-zp))-std::log((dnpn-zn)/(dnpp-zp))
         -std::log((dpnn-zn)/(dpnp-zp))+std::log((dppn-zn)/(dppp-zp));
    dm[5]=std::atan(zn*xn/(yn*dnnn))-std::atan(zp*xn/(yn*dnnp))
         -std::atan(zn*xp/(yn*dpnn))+std::atan(zp*xp/(yn*dpnp))
         -std::atan(zn*xn/(yp*dnpn))+std::atan(zp*xn/(yp*dnpp))
         +std::atan(zn*xp/(yp*dppn))-std::atan(zp*xp/(yp*dppp));
    dm[6]=std::log((dnnn-xn)/(dpnn-xp))-std::log((dnpn-xn)/(dppn-xp))
         -std::log((dnnp-xn)/(dpnp-xp))+std::log((dnpp-xn)/(dppp-xp));
    dm[7]=std::log((dnnn-yn)/(dnpn-yp))-std::log((dnnp-yn)/(dnpp-yp))
         -std::log((dpnn-yn)/(dppn-yp))+std::log((dpnp-yn)/(dppp-yp));
    dm[8]=std::log((dnnn-xn)/(dpnn-xp))-std::log((dnnp-xn)/(dpnp-xp))
         -std::log((dnpn-xn)/(dppn-xp))+std::log((dnpp-xn)/(dppp-xp));
    dm[9]=std::atan(xn*yn/(zn*dnnn))-std::atan(xp*yn/(zn*dpnn))
         -std::atan(xn*yp/(zn*dnpn))+std::atan(xp*yp/(zn*dppn))
         -std::atan(xn*yn/(zp*dnnp))+std::atan(xp*yn/(zp*dpnp))
         +std::atan(xn*yp/(zp*dnpp))-std::atan(xp*yp/(zp*dppp));
}

static void calt(double thik, int mdx, int mdy,
                 double taa[],double tab[],double tac[],
                 double tba[],double tbb[],double tbc[],
                 double tca[],double tcb[],double tcc[])
{
    int mdx2=mdx/2, mdy2=mdy/2;
    double a=0.49999, b=0.5*thik, dm[10];
    for(int j=0;j<mdy;j++) for(int i=0;i<mdx;i++){
        int ikn=j*mdx+i;
        taa[ikn]=tab[ikn]=tac[ikn]=tba[ikn]=tbb[ikn]=tbc[ikn]=
        tca[ikn]=tcb[ikn]=tcc[ikn]=0.0;
        for(int jy=-4;jy<=4;jy++){
            double sy=double(j-mdy2)+0.1*double(jy);
            for(int ix=-4;ix<=4;ix++){
                double sx=double(i-mdx2)+0.1*double(ix);
                ctt(b,a,sx,sy,dm);
                taa[ikn]+=dm[1];tab[ikn]+=dm[2];tac[ikn]+=dm[3];
                tba[ikn]+=dm[4];tbb[ikn]+=dm[5];tbc[ikn]+=dm[6];
                tca[ikn]+=dm[7];tcb[ikn]+=dm[8];tcc[ikn]+=dm[9];
            }
        }
        taa[ikn]/=81.;tab[ikn]/=81.;tac[ikn]/=81.;
        tba[ikn]/=81.;tbb[ikn]/=81.;tbc[ikn]/=81.;
        tca[ikn]/=81.;tcb[ikn]/=81.;tcc[ikn]/=81.;
    }
}

/* ── GPU kernels (v2 style) ── */

/* Pack one SoA component to complex device buffer (imag=0) */
__global__ static void pack_kernel(
    const double* __restrict__ y,
    cufftDoubleComplex* __restrict__ out,
    int comp, int ncell)
{
    int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=ncell) return;
    out[c].x = y[comp*ncell+c];
    out[c].y = 0.0;
}

/* Ĥ_α = Σ_β f̂_αβ · M̂_β  (all on device, f̂ is constant) */
__global__ static void multiply_kernel(
    const cufftDoubleComplex* __restrict__ Mx,
    const cufftDoubleComplex* __restrict__ My,
    const cufftDoubleComplex* __restrict__ Mz,
    const cufftDoubleComplex* __restrict__ faa,
    const cufftDoubleComplex* __restrict__ fab,
    const cufftDoubleComplex* __restrict__ fac,
    const cufftDoubleComplex* __restrict__ fba,
    const cufftDoubleComplex* __restrict__ fbb,
    const cufftDoubleComplex* __restrict__ fbc,
    const cufftDoubleComplex* __restrict__ fca,
    const cufftDoubleComplex* __restrict__ fcb,
    const cufftDoubleComplex* __restrict__ fcc,
    cufftDoubleComplex* __restrict__ Hx,
    cufftDoubleComplex* __restrict__ Hy,
    cufftDoubleComplex* __restrict__ Hz,
    int nk)
{
    int k=blockIdx.x*blockDim.x+threadIdx.x;
    if(k>=nk) return;
#define CR(A,B) ((A)[k].x*(B)[k].x-(A)[k].y*(B)[k].y)
#define CI(A,B) ((A)[k].x*(B)[k].y+(A)[k].y*(B)[k].x)
    Hx[k].x=CR(faa,Mx)+CR(fab,My)+CR(fac,Mz);
    Hx[k].y=CI(faa,Mx)+CI(fab,My)+CI(fac,Mz);
    Hy[k].x=CR(fba,Mx)+CR(fbb,My)+CR(fbc,Mz);
    Hy[k].y=CI(fba,Mx)+CI(fbb,My)+CI(fbc,Mz);
    Hz[k].x=CR(fca,Mx)+CR(fcb,My)+CR(fcc,Mz);
    Hz[k].y=CI(fca,Mx)+CI(fcb,My)+CI(fcc,Mz);
#undef CR
#undef CI
}

/*
 * scatter_kernel with professor's index remapping:
 *   jy = (j<nn2) ? (nn2-j) : (nn-j+nn2-1)
 *   ix = (i<nn2) ? (nn2-i) : (nn-i+nn2-1)
 *   h[j][i] += scale * hhat[jy][ix].x
 *
 * Results written to host array h_out (CPU) after D2H.
 * Here we write to a device array and D2H once at the end.
 */
__global__ static void scatter_kernel(
    const cufftDoubleComplex* __restrict__ hhat,
    double* __restrict__ out,    /* device, comp*ncell + cell */
    int comp, int ncell, int nx, int ny, double scale)
{
    int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    int j=cell/nx, i=cell%nx;
    int nx2=nx/2, ny2=ny/2;
    int jy=(j<ny2)?(ny2-j):(ny-j+ny2-1);
    int ix=(i<nx2)?(nx2-i):(nx-i+nx2-1);
    int idxnew=jy*nx+ix;
    if(idxnew<0||idxnew>=ncell) return;
    out[comp*ncell+cell] += scale * hhat[idxnew].x;
}

int main()
{
    const int nn  = NN;
    const int nn2 = nn/2;
    const int N   = nn*nn;
    const size_t csz = (size_t)N * sizeof(cufftDoubleComplex);
    const size_t rsz = (size_t)N * sizeof(double);

    /* ── step 1: calt on CPU ── */
    double *taa=new double[N],*tab=new double[N],*tac=new double[N];
    double *tba=new double[N],*tbb=new double[N],*tbc=new double[N];
    double *tca=new double[N],*tcb=new double[N],*tcc=new double[N];
    calt(THICK,nn,nn,taa,tab,tac,tba,tbb,tbc,tca,tcb,tcc);

    /* ── step 2: pack into complex (imag=0) ── */
    cufftDoubleComplex *htmp[9];
    double *tptrs[9]={taa,tab,tac,tba,tbb,tbc,tca,tcb,tcc};
    for(int c=0;c<9;c++){
        htmp[c]=new cufftDoubleComplex[N];
        for(int idx=0;idx<N;idx++){htmp[c][idx].x=tptrs[c][idx];htmp[c][idx].y=0.0;}
    }
    delete[]taa;delete[]tab;delete[]tac;delete[]tba;delete[]tbb;
    delete[]tbc;delete[]tca;delete[]tcb;delete[]tcc;

    /* ── step 3: cuFFT Z2Z plan ── */
    cufftHandle plan;
    if(cufftPlan2d(&plan,nn,nn,CUFFT_Z2Z)!=CUFFT_SUCCESS){
        std::printf("cufftPlan2d failed\n"); return 1;
    }

    /* ── steps 4-5: H2D + FFT → d_fhat (stays on device) ── */
    cufftDoubleComplex *d_fhat[9];
    for(int c=0;c<9;c++){
        cudaMalloc((void**)&d_fhat[c],csz);
        cudaMemcpy(d_fhat[c],htmp[c],csz,cudaMemcpyHostToDevice);
        delete[]htmp[c];
        cufftExecZ2Z(plan,d_fhat[c],d_fhat[c],CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();

    /* ── same magnetization as demag_test.cu ── */
    double *hma=new double[N],*hmb=new double[N],*hmc=new double[N];
    int i1=nn/3,i2=2*nn/3,j1=nn/3,j2=2*nn/3;
    for(int j=0;j<nn;j++) for(int i=0;i<nn;i++){
        int idx=j*nn+i;
        hma[idx]=1.0; hmb[idx]=0.0; hmc[idx]=0.0;
        if(i>i1&&i<i2&&j>j1&&j<j2) hma[idx]=0.0;
    }

    /* ── upload M to device as SoA [mx|my|mz] ── */
    double *d_y;   /* SoA: [mx|my|mz], size 3*N doubles */
    cudaMalloc((void**)&d_y, (size_t)3*N*sizeof(double));
    cudaMemcpy(d_y,          hma, rsz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y+N,        hmb, rsz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y+2*N,      hmc, rsz, cudaMemcpyHostToDevice);
    delete[]hma; delete[]hmb; delete[]hmc;

    /* ── allocate M̂ and Ĥ on device ── */
    cufftDoubleComplex *d_mhat[3], *d_hhat[3];
    for(int c=0;c<3;c++){
        cudaMalloc((void**)&d_mhat[c],csz);
        cudaMalloc((void**)&d_hhat[c],csz);
    }

    /* ── allocate h_out on device (SoA [hx|hy|hz], zeroed) ── */
    double *d_hout;
    cudaMalloc((void**)&d_hout,(size_t)3*N*sizeof(double));
    cudaMemset(d_hout,0,(size_t)3*N*sizeof(double));

    /* ── time the Apply ── */
    cudaEvent_t t0,t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);

    const int blk=256;
    const int g=(N+blk-1)/blk;
    const double scale = 1.0 / (double)N;  /* strength=1, 1/(nn*nn) */

    /* step 6: pack M → d_mhat (GPU) */
    for(int comp=0;comp<3;comp++)
        pack_kernel<<<g,blk>>>(d_y,d_mhat[comp],comp,N);

    /* step 7: FFT M̂ (in-place, GPU) */
    for(int comp=0;comp<3;comp++)
        cufftExecZ2Z(plan,d_mhat[comp],d_mhat[comp],CUFFT_FORWARD);

    /* step 8: Ĥ = f̂ · M̂  (GPU) */
    multiply_kernel<<<g,blk>>>(
        d_mhat[0],d_mhat[1],d_mhat[2],
        d_fhat[0],d_fhat[1],d_fhat[2],
        d_fhat[3],d_fhat[4],d_fhat[5],
        d_fhat[6],d_fhat[7],d_fhat[8],
        d_hhat[0],d_hhat[1],d_hhat[2],
        N);

    /* step 9: IFFT Ĥ (in-place, GPU) */
    for(int comp=0;comp<3;comp++)
        cufftExecZ2Z(plan,d_hhat[comp],d_hhat[comp],CUFFT_INVERSE);

    /* step 10: scatter with professor's index remapping (GPU) */
    for(int comp=0;comp<3;comp++)
        scatter_kernel<<<g,blk>>>(
            d_hhat[comp],d_hout,comp,N,nn,nn,scale);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float ms=0.f;
    cudaEventElapsedTime(&ms,t0,t1);
    std::fprintf(stderr,"[v2] nn=%d  Apply time = %.3f ms\n",nn,ms);

    /* ── D2H h_out ── */
    double *hrha=new double[N],*hrhb=new double[N],*hrhc=new double[N];
    cudaMemcpy(hrha,d_hout,      rsz,cudaMemcpyDeviceToHost);
    cudaMemcpy(hrhb,d_hout+N,   rsz,cudaMemcpyDeviceToHost);
    cudaMemcpy(hrhc,d_hout+2*N, rsz,cudaMemcpyDeviceToHost);

    /* ── print with SAME format as demag_test.cu ──
     *
     * demag_test.cu output loop:
     *   for j: jy = (j<nn2)?(nn2-j):(nn-j+nn2-1)
     *   for i: ix = (i<nn2)?(nn2-i):(nn-i+nn2-1)
     *   idxnew = jy*nn+ix
     *   printf("%f %f %f\n", hrha[idxnew]/(nn*nn), ...)
     *
     * In v2 the scatter_kernel already applied the remapping and the
     * 1/(nn*nn) normalization, so h_out[j*nn+i] is already the correct
     * h(i,j).  We just print in (j,i) order.
     *
     * To match demag_test.cu's output ordering exactly we apply the
     * same outer loop (j,i) with the same remapping on the OUTPUT index:
     */
    std::printf("%d %d %d\n",nn,nn,nn);
    for(int j=0;j<nn;j++){
        int jy=(j<nn2)?(nn2-j):(nn-j+nn2-1);
        for(int i=0;i<nn;i++){
            int ix=(i<nn2)?(nn2-i):(nn-i+nn2-1);
            int idxnew=jy*nn+ix;
            /* scatter_kernel already wrote h[comp*N + (j*nn+i)] = h(i,j)
             * so we read by the same (j,i) index, no extra remapping needed */
            int idx=j*nn+i;
            std::printf("%f %f %f\n", hrha[idx], hrhb[idx], hrhc[idx]);
        }
    }

    /* ── cleanup ── */
    cufftDestroy(plan);
    for(int c=0;c<9;c++) cudaFree(d_fhat[c]);
    for(int c=0;c<3;c++){cudaFree(d_mhat[c]);cudaFree(d_hhat[c]);}
    cudaFree(d_y); cudaFree(d_hout);
    delete[]hrha; delete[]hrhb; delete[]hrhc;
    return 0;
}
