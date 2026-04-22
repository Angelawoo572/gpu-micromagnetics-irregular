/*
<<<<<<< Updated upstream
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
=======
 * demag_fft_v2_test.cu
 *
 * Standalone test: verify that the optimized version (f̂ stays on device,
 * multiply on GPU) gives IDENTICAL results to demag_test.cu.
 *
 * Key optimization:
 *   OLD: hofaa..hofcc live on HOST, multiply loop on CPU, then H2D back
 *   NEW: dofaa..dofcc live on DEVICE permanently, multiply kernel on GPU
 *        → eliminates 9 D2H + 9 H2D transfers per f() call
 *        → f̂ computed once at Init, never touched again
 *
 * Compile:
 *   nvcc -O3 -arch=sm_89 demag_fft_v2_test.cu -lcufft -o demag_fft_v2_test
 *
 * Run and compare with demag_test.cu:
 *   ./demag_test         > out_v1.txt
 *   ./demag_fft_v2_test  > out_v2.txt
 *   diff out_v1.txt out_v2.txt        # should be empty
 *
 * In MATLAB:
 *   A = load('out_v1.txt');  B = load('out_v2.txt');
 *   max(max(abs(A(2:end,:) - B(2:end,:))))   % should be ~0 or machine epsilon
>>>>>>> Stashed changes
 */

#include <cstdio>
#include <cmath>
#include <cufft.h>
#include <cuda_runtime.h>

<<<<<<< Updated upstream
/* ── ctt: exact copy from  pseudocode (1-indexed dm[]) ── */
=======
static const int NN    = 64;
static const double THICK = 100.0;

/* ── cadd / cmul (same as demag_test.cu) ── */
static inline cufftDoubleComplex cadd(cufftDoubleComplex a, cufftDoubleComplex b)
{ cufftDoubleComplex r; r.x=a.x+b.x; r.y=a.y+b.y; return r; }

static inline cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b)
{ cufftDoubleComplex r; r.x=a.x*b.x-a.y*b.y; r.y=a.x*b.y+a.y*b.x; return r; }

/* ── ctt / calt: verbatim from professor's pseudocode (1-indexed dm[]) ── */
>>>>>>> Stashed changes
static void ctt(double b, double a, double sx, double sy, double dm[])
{
    double sz=0.0;
    double xn=sx-a,xp=sx+a,yn=sy-a,yp=sy+a,zn=sz-b,zp=sz+b;
    double xn2=xn*xn,xp2=xp*xp,yn2=yn*yn,yp2=yp*yp,zn2=zn*zn,zp2=zp*zp;
    double dnnn=std::sqrt(xn2+yn2+zn2),dpnn=std::sqrt(xp2+yn2+zn2);
    double dnpn=std::sqrt(xn2+yp2+zn2),dnnp=std::sqrt(xn2+yn2+zp2);
    double dppn=std::sqrt(xp2+yp2+zn2),dnpp=std::sqrt(xn2+yp2+zp2);
    double dpnp=std::sqrt(xp2+yn2+zp2),dppp=std::sqrt(xp2+yp2+zp2);
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
static void calt(double thik, int mdx, int mdy,
                 double taa[],double tab[],double tac[],
                 double tba[],double tbb[],double tbc[],
                 double tca[],double tcb[],double tcc[])
{
    int mdx2=mdx/2, mdy2=mdy/2;
    double a=0.49999, b=0.5*thik;
    double dm[10];
    for(int j=0;j<mdy;j++) for(int i=0;i<mdx;i++){
        int ikn=j*mdx+i;
        taa[ikn]=tab[ikn]=tac[ikn]=0.0;
        tba[ikn]=tbb[ikn]=tbc[ikn]=0.0;
>>>>>>> Stashed changes
        tca[ikn]=tcb[ikn]=tcc[ikn]=0.0;
        for(int jy=-4;jy<=4;jy++){
            double sy=double(j-mdy2)+0.1*double(jy);
            for(int ix=-4;ix<=4;ix++){
                double sx=double(i-mdx2)+0.1*double(ix);
                ctt(b,a,sx,sy,dm);
<<<<<<< Updated upstream
                taa[ikn]+=dm[1];tab[ikn]+=dm[2];tac[ikn]+=dm[3];
                tba[ikn]+=dm[4];tbb[ikn]+=dm[5];tbc[ikn]+=dm[6];
                tca[ikn]+=dm[7];tcb[ikn]+=dm[8];tcc[ikn]+=dm[9];
            }
        }
        taa[ikn]/=81.;tab[ikn]/=81.;tac[ikn]/=81.;
        tba[ikn]/=81.;tbb[ikn]/=81.;tbc[ikn]/=81.;
        tca[ikn]/=81.;tcb[ikn]/=81.;tcc[ikn]/=81.;
=======
                taa[ikn]+=dm[1]; tab[ikn]+=dm[2]; tac[ikn]+=dm[3];
                tba[ikn]+=dm[4]; tbb[ikn]+=dm[5]; tbc[ikn]+=dm[6];
                tca[ikn]+=dm[7]; tcb[ikn]+=dm[8]; tcc[ikn]+=dm[9];
            }
        }
        taa[ikn]/=81.; tab[ikn]/=81.; tac[ikn]/=81.;
        tba[ikn]/=81.; tbb[ikn]/=81.; tbc[ikn]/=81.;
        tca[ikn]/=81.; tcb[ikn]/=81.; tcc[ikn]/=81.;
>>>>>>> Stashed changes
    }
}

/* 
<<<<<<< Updated upstream
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
=======
 * GPU kernel: pointwise multiply  ĥ = f̂ · m̂  (all on device)
 *
 * f̂ (tensor spectra) is PERMANENT on device — computed once at init.
 * m̂ (magnetization spectrum) is computed fresh each call.
 *
 * hkha[k] = hofaa[k]*homa[k] + hofab[k]*homb[k] + hofac[k]*homc[k]
 * (same formula as professor's CPU loop, but on GPU)
 *  */
__global__ static void multiply_kernel(
    /* f̂: tensor spectra (permanent device arrays) */
    const cufftDoubleComplex* __restrict__ dofaa,
    const cufftDoubleComplex* __restrict__ dofab,
    const cufftDoubleComplex* __restrict__ dofac,
    const cufftDoubleComplex* __restrict__ dofba,
    const cufftDoubleComplex* __restrict__ dofbb,
    const cufftDoubleComplex* __restrict__ dofbc,
    const cufftDoubleComplex* __restrict__ dofca,
    const cufftDoubleComplex* __restrict__ dofcb,
    const cufftDoubleComplex* __restrict__ dofcc,
    /* m̂: magnetization spectra (computed this call) */
    const cufftDoubleComplex* __restrict__ doma,
    const cufftDoubleComplex* __restrict__ domb,
    const cufftDoubleComplex* __restrict__ domc,
    /* ĥ: output field spectra */
    cufftDoubleComplex* __restrict__ dkha,
    cufftDoubleComplex* __restrict__ dkhb,
    cufftDoubleComplex* __restrict__ dkhc,
    int N)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= N) return;

    /* Ĥa = f̂aa·M̂a + f̂ab·M̂b + f̂ac·M̂c */
    cufftDoubleComplex ha = {0.0, 0.0};
    {
        cufftDoubleComplex t;
        /* f̂aa * M̂a */
        t.x = dofaa[k].x*doma[k].x - dofaa[k].y*doma[k].y;
        t.y = dofaa[k].x*doma[k].y + dofaa[k].y*doma[k].x;
        ha.x += t.x; ha.y += t.y;
        /* f̂ab * M̂b */
        t.x = dofab[k].x*domb[k].x - dofab[k].y*domb[k].y;
        t.y = dofab[k].x*domb[k].y + dofab[k].y*domb[k].x;
        ha.x += t.x; ha.y += t.y;
        /* f̂ac * M̂c */
        t.x = dofac[k].x*domc[k].x - dofac[k].y*domc[k].y;
        t.y = dofac[k].x*domc[k].y + dofac[k].y*domc[k].x;
        ha.x += t.x; ha.y += t.y;
>>>>>>> Stashed changes
    }
    dkha[k] = ha;

<<<<<<< Updated upstream
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
=======
    /* Ĥb = f̂ba·M̂a + f̂bb·M̂b + f̂bc·M̂c */
    cufftDoubleComplex hb = {0.0, 0.0};
    {
        cufftDoubleComplex t;
        t.x = dofba[k].x*doma[k].x - dofba[k].y*doma[k].y;
        t.y = dofba[k].x*doma[k].y + dofba[k].y*doma[k].x;
        hb.x += t.x; hb.y += t.y;
        t.x = dofbb[k].x*domb[k].x - dofbb[k].y*domb[k].y;
        t.y = dofbb[k].x*domb[k].y + dofbb[k].y*domb[k].x;
        hb.x += t.x; hb.y += t.y;
        t.x = dofbc[k].x*domc[k].x - dofbc[k].y*domc[k].y;
        t.y = dofbc[k].x*domc[k].y + dofbc[k].y*domc[k].x;
        hb.x += t.x; hb.y += t.y;
    }
    dkhb[k] = hb;

    /* Ĥc = f̂ca·M̂a + f̂cb·M̂b + f̂cc·M̂c */
    cufftDoubleComplex hc = {0.0, 0.0};
    {
        cufftDoubleComplex t;
        t.x = dofca[k].x*doma[k].x - dofca[k].y*doma[k].y;
        t.y = dofca[k].x*doma[k].y + dofca[k].y*doma[k].x;
        hc.x += t.x; hc.y += t.y;
        t.x = dofcb[k].x*domb[k].x - dofcb[k].y*domb[k].y;
        t.y = dofcb[k].x*domb[k].y + dofcb[k].y*domb[k].x;
        hc.x += t.x; hc.y += t.y;
        t.x = dofcc[k].x*domc[k].x - dofcc[k].y*domc[k].y;
        t.y = dofcc[k].x*domc[k].y + dofcc[k].y*domc[k].x;
        hc.x += t.x; hc.y += t.y;
    }
    dkhc[k] = hc;
}

int main()
{
    const int nn  = NN;
    const int nn2 = nn / 2;
    const int N   = nn * nn;
    const size_t csz = (size_t)N * sizeof(cufftDoubleComplex);

    /* ── Step 1: calt on CPU (same as demag_test.cu) ── */
    double *faa=new double[N],*fab=new double[N],*fac=new double[N];
    double *fba=new double[N],*fbb=new double[N],*fbc=new double[N];
    double *fca=new double[N],*fcb=new double[N],*fcc=new double[N];

    calt(THICK,nn,nn, faa,fab,fac, fba,fbb,fbc, fca,fcb,fcc);

    /* ── Step 2: pack tensor into complex host arrays (imag=0) ── */
    cufftDoubleComplex *hfaa=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfab=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfac=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfba=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfbb=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfbc=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfca=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfcb=new cufftDoubleComplex[N];
    cufftDoubleComplex *hfcc=new cufftDoubleComplex[N];

    for(int idx=0;idx<N;idx++){
        hfaa[idx].x=faa[idx]; hfaa[idx].y=0.0;
        hfab[idx].x=fab[idx]; hfab[idx].y=0.0;
        hfac[idx].x=fac[idx]; hfac[idx].y=0.0;
        hfba[idx].x=fba[idx]; hfba[idx].y=0.0;
        hfbb[idx].x=fbb[idx]; hfbb[idx].y=0.0;
        hfbc[idx].x=fbc[idx]; hfbc[idx].y=0.0;
        hfca[idx].x=fca[idx]; hfca[idx].y=0.0;
        hfcb[idx].x=fcb[idx]; hfcb[idx].y=0.0;
        hfcc[idx].x=fcc[idx]; hfcc[idx].y=0.0;
    }
    delete[]faa;delete[]fab;delete[]fac;
    delete[]fba;delete[]fbb;delete[]fbc;
    delete[]fca;delete[]fcb;delete[]fcc;

    /* ── Step 3: H2D tensor, FFT it, KEEP f̂ on device permanently ── */
    /* dofaa..dofcc are the PERMANENT device arrays for f̂ */
    cufftDoubleComplex *difaa,*difab,*difac,*difba,*difbb,*difbc,*difca,*difcb,*difcc;
    cufftDoubleComplex *dofaa,*dofab,*dofac,*dofba,*dofbb,*dofbc,*dofca,*dofcb,*dofcc;
>>>>>>> Stashed changes

    /* allocate M̂ and Ĥ (reused every timestep) */
    for(int c=0;c<3;c++){
        cudaMalloc((void**)&d->d_mhat[c],csz);
        cudaMalloc((void**)&d->d_hhat[c],csz);
    }

<<<<<<< Updated upstream
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
=======
    cudaMemcpy(difaa,hfaa,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difab,hfab,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difac,hfac,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difba,hfba,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbb,hfbb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difbc,hfbc,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difca,hfca,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcb,hfcb,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(difcc,hfcc,csz,cudaMemcpyHostToDevice);

    delete[]hfaa;delete[]hfab;delete[]hfac;
    delete[]hfba;delete[]hfbb;delete[]hfbc;
    delete[]hfca;delete[]hfcb;delete[]hfcc;

    cufftHandle plan;
    cufftPlan2d(&plan, nn, nn, CUFFT_Z2Z);

    /* FFT the tensor — dofaa..dofcc now hold f̂, permanently on device */
    cufftExecZ2Z(plan,difaa,dofaa,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difab,dofab,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difac,dofac,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difba,dofba,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difbb,dofbb,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difbc,dofbc,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difca,dofca,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difcb,dofcb,CUFFT_FORWARD);
    cufftExecZ2Z(plan,difcc,dofcc,CUFFT_FORWARD);
    cudaDeviceSynchronize();

    /* free the input-side tensor buffers (no longer needed) */
    cudaFree(difaa);cudaFree(difab);cudaFree(difac);
    cudaFree(difba);cudaFree(difbb);cudaFree(difbc);
    cudaFree(difca);cudaFree(difcb);cudaFree(difcc);

    /* ── Step 4: same magnetization pattern as demag_test.cu ── */
    cufftDoubleComplex *hma_c=new cufftDoubleComplex[N];
    cufftDoubleComplex *hmb_c=new cufftDoubleComplex[N];
    cufftDoubleComplex *hmc_c=new cufftDoubleComplex[N];

    int i1=nn/3, i2=2*nn/3, j1=nn/3, j2=2*nn/3;
    for(int j=0;j<nn;j++) for(int i=0;i<nn;i++){
        int idx=j*nn+i;
        double ma=1.0;
        if(i>i1&&i<i2&&j>j1&&j<j2) ma=0.0;
        hma_c[idx].x=ma;  hma_c[idx].y=0.0;
        hmb_c[idx].x=0.0; hmb_c[idx].y=0.0;
        hmc_c[idx].x=0.0; hmc_c[idx].y=0.0;
    }

    /* ── Step 5: H2D magnetization, FFT → m̂ stays on device ── */
    cufftDoubleComplex *dima,*dimb,*dimc;
    cufftDoubleComplex *doma,*domb,*domc;
    cudaMalloc((void**)&dima,csz); cudaMalloc((void**)&doma,csz);
    cudaMalloc((void**)&dimb,csz); cudaMalloc((void**)&domb,csz);
    cudaMalloc((void**)&dimc,csz); cudaMalloc((void**)&domc,csz);

    cudaMemcpy(dima,hma_c,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(dimb,hmb_c,csz,cudaMemcpyHostToDevice);
    cudaMemcpy(dimc,hmc_c,csz,cudaMemcpyHostToDevice);
    delete[]hma_c; delete[]hmb_c; delete[]hmc_c;

    cufftExecZ2Z(plan,dima,doma,CUFFT_FORWARD);
    cufftExecZ2Z(plan,dimb,domb,CUFFT_FORWARD);
    cufftExecZ2Z(plan,dimc,domc,CUFFT_FORWARD);
    /* doma/domb/domc now hold m̂, all on device — no D2H needed */

    /* ── Step 6: GPU multiply  ĥ = f̂ · m̂  (everything on device) ── */
    /* This replaces the CPU cadd/cmul loop from demag_test.cu        */
    cufftDoubleComplex *dkha,*dkhb,*dkhc;
    cufftDoubleComplex *drha,*drhb,*drhc;
    cudaMalloc((void**)&dkha,csz); cudaMalloc((void**)&drha,csz);
    cudaMalloc((void**)&dkhb,csz); cudaMalloc((void**)&drhb,csz);
    cudaMalloc((void**)&dkhc,csz); cudaMalloc((void**)&drhc,csz);

    {
        int block=256, grid=(N+block-1)/block;
        multiply_kernel<<<grid,block>>>(
            dofaa,dofab,dofac,
            dofba,dofbb,dofbc,
            dofca,dofcb,dofcc,
            doma,domb,domc,
            dkha,dkhb,dkhc,
            N);
        cudaDeviceSynchronize();
    }

    /* ── Step 7: IFFT ĥ → h (on device) ── */
    cufftExecZ2Z(plan,dkha,drha,CUFFT_INVERSE);
    cufftExecZ2Z(plan,dkhb,drhb,CUFFT_INVERSE);
    cufftExecZ2Z(plan,dkhc,drhc,CUFFT_INVERSE);
    cudaDeviceSynchronize();

    /* ── Step 8: D2H result ── */
    cufftDoubleComplex *hrha=new cufftDoubleComplex[N];
    cufftDoubleComplex *hrhb=new cufftDoubleComplex[N];
    cufftDoubleComplex *hrhc=new cufftDoubleComplex[N];
    cudaMemcpy(hrha,drha,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(hrhb,drhb,csz,cudaMemcpyDeviceToHost);
    cudaMemcpy(hrhc,drhc,csz,cudaMemcpyDeviceToHost);

    /* ── Step 9: print with EXACT same index remapping as demag_test.cu ── */
    std::printf("%d %d %d\n", nn, nn, nn);
    for(int j=0;j<nn;j++){
        int jy = (j<nn2) ? (nn2-j) : (nn-j+nn2-1);
        for(int i=0;i<nn;i++){
            int ix = (i<nn2) ? (nn2-i) : (nn-i+nn2-1);
            int idxnew = jy*nn + ix;
            double hx = hrha[idxnew].x / (double)(nn*nn);
            double hy = hrhb[idxnew].x / (double)(nn*nn);
            double hz = hrhc[idxnew].x / (double)(nn*nn);
            std::printf("%f %f %f\n", hx, hy, hz);
        }
    }

    /* cleanup */
    cufftDestroy(plan);
    cudaFree(dofaa);cudaFree(dofab);cudaFree(dofac);
    cudaFree(dofba);cudaFree(dofbb);cudaFree(dofbc);
    cudaFree(dofca);cudaFree(dofcb);cudaFree(dofcc);
    cudaFree(dima);cudaFree(dimb);cudaFree(dimc);
    cudaFree(doma);cudaFree(domb);cudaFree(domc);
    cudaFree(dkha);cudaFree(dkhb);cudaFree(dkhc);
    cudaFree(drha);cudaFree(drhb);cudaFree(drhc);
    delete[]hrha;delete[]hrhb;delete[]hrhc;

    return 0;
>>>>>>> Stashed changes
}
