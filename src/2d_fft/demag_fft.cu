/*
 * demag_fft.cu
 *
 * Key fix over v1:
 *   f̂ (FFT of demag tensor) is computed ONCE at Init and stays on device.
 *   Demag_Apply does everything on GPU — no D2H/H2D round-trips for
 *   the multiply step.
 *
 * v1 (inefficient) flow in Apply:
 *   D2H y → pack CPU → H2D → FFT GPU → D2H M̂ → multiply CPU → H2D Ĥ
 *   → IFFT GPU → D2H h → scatter CPU → H2D h_out
 *
 * v2 (efficient) flow in Apply — all GPU:
 *   pack_kernel(y_dev→d_mhat) → FFT(d_mhat) → multiply_kernel(d_fhat·d_mhat→d_hhat)
 *   → IFFT(d_hhat) → scatter_kernel(d_hhat→h_out)
 *
 * f̂ = d_fhat[9] never leaves device after Init.
 */

#include "demag_fft.h"

#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ── ctt: exact copy from  pseudocode (1-indexed dm[]) ── */
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

/* ── calt: exact copy (1-indexed dm[]) ── */
static int calt(double thik,int mdx,int mdy,
                double taa[],double tab[],double tac[],
                double tba[],double tbb[],double tbc[],
                double tca[],double tcb[],double tcc[])
{
    int mdx2=mdx/2,mdy2=mdy/2;
    double a=0.49999,b=0.5*thik,dm[10];
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
    return 1;
}

/* 
 * GPU kernels
 *  */

/* Pack one SoA real component → complex device buffer (imag=0) */
__global__ static void pack_real_to_complex_kernel(
    const double* __restrict__ y_dev,
    cufftDoubleComplex* __restrict__ out,
    int comp, int ncell)
{
    int cell = blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=ncell) return;
    out[cell].x = y_dev[comp*ncell+cell];
    out[cell].y = 0.0;
}

/* Pointwise multiply: Ĥ_α = Σ_β f̂_αβ · M̂_β  (all on device) */
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

/* Scatter IFFT result into h_out SoA with  index remapping */
__global__ static void scatter_add_kernel(
    const cufftDoubleComplex* __restrict__ hhat,
    double* __restrict__ h_out,
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
    h_out[comp*ncell+cell] += scale * hhat[idxnew].x;
}

/* 
 * DemagData
 *  */
struct DemagData {
    int nx, ny, ncell, nk;
    double strength;
    cufftHandle plan;
    cufftDoubleComplex *d_fhat[9];  /* constant, on device forever */
    cufftDoubleComplex *d_mhat[3];  /* M̂, updated each timestep */
    cufftDoubleComplex *d_hhat[3];  /* Ĥ, updated each timestep */
};

/* 
 * Demag_Init
 *  */
DemagData* Demag_Init(int nx, int ny, double thick, double demag_strength)
{
    DemagData *d=(DemagData*)calloc(1,sizeof(DemagData));
    if(!d){fprintf(stderr,"[demag v2] calloc failed\n");return NULL;}
    d->nx=nx; d->ny=ny; d->ncell=nx*ny; d->nk=nx*ny;
    d->strength=demag_strength;

    const size_t csz=(size_t)(nx*ny)*sizeof(cufftDoubleComplex);

    /* step 1: calt on CPU */
    double *t[9];
    for(int c=0;c<9;c++){
        t[c]=(double*)calloc(nx*ny,sizeof(double));
        if(!t[c]){
            fprintf(stderr,"[demag v2] tensor alloc failed\n");
            for(int cc=0;cc<c;cc++) free(t[cc]);
            free(d); return NULL;
        }
    }
    calt(thick,nx,ny,t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8]);

    /* step 2: pack into complex on CPU */
    cufftDoubleComplex *htmp[9];
    for(int c=0;c<9;c++){
        htmp[c]=(cufftDoubleComplex*)malloc(csz);
        if(!htmp[c]){
            fprintf(stderr,"[demag v2] htmp alloc failed\n");
            for(int cc=0;cc<c;cc++) free(htmp[cc]);
            for(int cc=0;cc<9;cc++) free(t[cc]);
            free(d); return NULL;
        }
        for(int idx=0;idx<nx*ny;idx++){
            htmp[c][idx].x=t[c][idx];
            htmp[c][idx].y=0.0;
        }
        free(t[c]);
    }

    /* step 3: cuFFT Z2Z plan */
    {
        cufftResult r=cufftPlan2d(&d->plan,ny,nx,CUFFT_Z2Z);
        if(r!=CUFFT_SUCCESS){
            fprintf(stderr,"[demag v2] cufftPlan2d failed: %d\n",(int)r);
            for(int c=0;c<9;c++) free(htmp[c]);
            free(d); return NULL;
        }
    }

    /* steps 4-5: H2D + FFT → d_fhat — stays on device permanently */
    for(int c=0;c<9;c++){
        if(cudaMalloc((void**)&d->d_fhat[c],csz)!=cudaSuccess){
            fprintf(stderr,"[demag v2] cudaMalloc d_fhat[%d] failed\n",c);
            for(int cc=0;cc<c;cc++) cudaFree(d->d_fhat[cc]);
            for(int cc=0;cc<9;cc++) free(htmp[cc]);
            cufftDestroy(d->plan); free(d); return NULL;
        }
        cudaMemcpy(d->d_fhat[c],htmp[c],csz,cudaMemcpyHostToDevice);
        free(htmp[c]);
        cufftExecZ2Z(d->plan,d->d_fhat[c],d->d_fhat[c],CUFFT_FORWARD);
    }
    cudaDeviceSynchronize();

    /* allocate M̂ and Ĥ (reused every timestep) */
    for(int c=0;c<3;c++){
        cudaMalloc((void**)&d->d_mhat[c],csz);
        cudaMalloc((void**)&d->d_hhat[c],csz);
    }

    return d;
}

/* 
 * Demag_Apply  (all GPU, no D2H/H2D in hot path)
 *  */
void Demag_Apply(DemagData *d, const double *y_dev, double *h_out)
{
    if(!d) return;
    const int ncell=d->ncell, nk=d->nk, nx=d->nx, ny=d->ny;
    const double scale=d->strength/(double)(nx*ny);
    const int blk=256;
    const int g_cell=(ncell+blk-1)/blk;
    const int g_k   =(nk   +blk-1)/blk;

    /* step 6: pack M → d_mhat (GPU kernel, no D2H) */
    for(int comp=0;comp<3;comp++)
        pack_real_to_complex_kernel<<<g_cell,blk>>>(
            y_dev, d->d_mhat[comp], comp, ncell);

    /* step 7: FFT M̂ (in-place, GPU) */
    for(int comp=0;comp<3;comp++)
        cufftExecZ2Z(d->plan, d->d_mhat[comp],
                     d->d_mhat[comp], CUFFT_FORWARD);

    /* step 8: Ĥ = f̂ · M̂  (GPU kernel, f̂ is constant on device) */
    multiply_kernel<<<g_k,blk>>>(
        d->d_mhat[0], d->d_mhat[1], d->d_mhat[2],
        d->d_fhat[0], d->d_fhat[1], d->d_fhat[2],
        d->d_fhat[3], d->d_fhat[4], d->d_fhat[5],
        d->d_fhat[6], d->d_fhat[7], d->d_fhat[8],
        d->d_hhat[0], d->d_hhat[1], d->d_hhat[2],
        nk);

    /* step 9: IFFT Ĥ (in-place, GPU) */
    for(int comp=0;comp<3;comp++)
        cufftExecZ2Z(d->plan, d->d_hhat[comp],
                     d->d_hhat[comp], CUFFT_INVERSE);

    /* step 10: scatter into h_out SoA (GPU kernel,  index remap) */
    for(int comp=0;comp<3;comp++)
        scatter_add_kernel<<<g_cell,blk>>>(
            d->d_hhat[comp], h_out, comp, ncell, nx, ny, scale);
}

/* 
 * Demag_Destroy
 *  */
void Demag_Destroy(DemagData *d)
{
    if(!d) return;
    cufftDestroy(d->plan);
    for(int c=0;c<9;c++) if(d->d_fhat[c]) cudaFree(d->d_fhat[c]);
    for(int c=0;c<3;c++){
        if(d->d_mhat[c]) cudaFree(d->d_mhat[c]);
        if(d->d_hhat[c]) cudaFree(d->d_hhat[c]);
    }
    free(d);
}
