/*==========================================================================
 * Testbench for NTT kernel — direct + four-step paths
 *
 * NOTE: psi_powers is non-const because the kernel reuses it as scratch
 * for the four-step transpose. The testbench re-generates psi_powers
 * before each test to ensure it's fresh.
 *==========================================================================*/

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>

#include "ntt.hpp"

static inline uint32_t ref_mod_mul(uint32_t a, uint32_t b, uint32_t q) {
    return (uint32_t)(((uint64_t)a * b) % q);
}
static inline uint32_t ref_mod_add(uint32_t a, uint32_t b, uint32_t q) {
    uint64_t s = (uint64_t)a + b; return (uint32_t)(s >= q ? s - q : s);
}
static inline uint32_t ref_mod_sub(uint32_t a, uint32_t b, uint32_t q) {
    return (a >= b) ? (a - b) : (a + q - b);
}
static uint32_t power_mod(uint32_t base, uint64_t exp, uint32_t mod) {
    uint64_t r = 1, b = base % mod;
    while (exp > 0) { if (exp & 1) r = (r * b) % mod; b = (b * b) % mod; exp >>= 1; }
    return (uint32_t)r;
}
static bool is_prime(uint64_t n) {
    if (n < 2) return false; if (n < 4) return true;
    if (n%2==0||n%3==0) return false;
    for (uint64_t d=5;d*d<=n;d+=6) if(n%d==0||n%(d+2)==0) return false;
    return true;
}
static uint32_t generate_ntt_modulus(uint32_t N, int bl=31) {
    uint64_t step=2*(uint64_t)N, lim=((uint64_t)1<<bl)-1;
    uint64_t k=(lim-1)/step, c=k*step+1;
    while(c>=2){if(is_prime(c))return(uint32_t)c;c-=step;} return 0;
}
static uint32_t find_generator(uint32_t q) {
    uint32_t phi=q-1; std::vector<uint32_t> facs; uint64_t x=phi;
    for(uint64_t d=2;d*d<=x;d++){if(x%d==0){facs.push_back(d);while(x%d==0)x/=d;}}
    if(x>1)facs.push_back((uint32_t)x);
    for(uint32_t g=2;g<q;g++){bool ok=true;for(auto f:facs)if(power_mod(g,phi/f,q)==1){ok=false;break;}if(ok)return g;} return 0;
}
static uint32_t find_psi(uint32_t N, uint32_t q) {
    return power_mod(find_generator(q),(q-1)/(2*N),q);
}
static void make_stockham_tw(uint32_t M, uint32_t omega_M, uint32_t q,
                             std::vector<uint32_t> &tw) {
    uint32_t logM=0; for(uint32_t t=M;t>1;t>>=1)logM++;
    tw.assign(M, 1);
    for(uint32_t s=0;s<logM;s++){
        uint32_t span=1u<<s, stride=M/(2*span);
        uint32_t step=power_mod(omega_M, stride, q), cur=1;
        for(uint32_t j=0;j<span;j++){tw[span+j]=cur;cur=ref_mod_mul(cur,step,q);}
    }
}

/* Direct-path tables */
static void make_direct_tables(uint32_t N, uint32_t q, uint32_t psi,
    std::vector<uint32_t>&pp, std::vector<uint32_t>&tw) {
    uint32_t omega=ref_mod_mul(psi,psi,q);
    pp.resize(N); pp[0]=1;
    for(uint32_t i=1;i<N;i++) pp[i]=ref_mod_mul(pp[i-1],psi,q);
    make_stockham_tw(N, omega, q, tw);
}

/* Four-step tables: [tw_col(N2) | tw_row(N1) | inter(N)] */
static void make_four_step_tables(uint32_t N, uint32_t q, uint32_t psi,
    std::vector<uint32_t>&pp, std::vector<uint32_t>&tw_packed) {
    uint32_t logN=0; for(uint32_t t=N;t>1;t>>=1) logN++;
    uint32_t logN1=logN>>1, logN2=logN-logN1;
    uint32_t N1=1u<<logN1, N2=1u<<logN2;
    uint32_t omega=ref_mod_mul(psi,psi,q);

    pp.resize(N); pp[0]=1;
    for(uint32_t i=1;i<N;i++) pp[i]=ref_mod_mul(pp[i-1],psi,q);

    uint32_t omega_2 = power_mod(omega, N1, q);  // N2-th root
    std::vector<uint32_t> tw_col;
    make_stockham_tw(N2, omega_2, q, tw_col);

    uint32_t omega_1 = power_mod(omega, N2, q);  // N1-th root
    std::vector<uint32_t> tw_row;
    make_stockham_tw(N1, omega_1, q, tw_row);

    // Inter-stage: omega^(col*row) stored at [row*N1+col]
    std::vector<uint32_t> inter(N);
    for(uint32_t row=0;row<N2;row++)
        for(uint32_t col=0;col<N1;col++)
            inter[row*N1+col]=power_mod(omega,((uint64_t)col*row)%N,q);

    tw_packed.resize(N2 + N1 + N);
    for(uint32_t i=0;i<N2;i++) tw_packed[i]=tw_col[i];
    for(uint32_t i=0;i<N1;i++) tw_packed[N2+i]=tw_row[i];
    for(uint32_t i=0;i<N;i++) tw_packed[N2+N1+i]=inter[i];
}

/* Reference NTT */
static uint32_t rev_bits(uint32_t x, uint32_t logN) {
    uint32_t r=0;for(uint32_t i=0;i<logN;i++){r=(r<<1)|(x&1);x>>=1;}return r;
}
static void ref_ntt(std::vector<uint32_t>&a, const std::vector<uint32_t>&pp, uint32_t q) {
    uint32_t N=(uint32_t)a.size(), logN=0; for(uint32_t t=N;t>1;t>>=1)logN++;
    for(uint32_t i=0;i<N;i++) a[i]=ref_mod_mul(a[i],pp[i],q);
    for(uint32_t i=0;i<N;i++){uint32_t j=rev_bits(i,logN);if(j>i)std::swap(a[i],a[j]);}
    uint32_t omega=ref_mod_mul(pp[1],pp[1],q);
    for(uint32_t s=0;s<logN;s++){uint32_t span=1u<<s,span2=span<<1,stride=N/(2*span),ws=power_mod(omega,stride,q);
    for(uint32_t k=0;k<N;k+=span2){uint32_t w=1;for(uint32_t j=0;j<span;j++){
    uint32_t u=a[k+j],v=ref_mod_mul(a[k+j+span],w,q);a[k+j]=ref_mod_add(u,v,q);a[k+j+span]=ref_mod_sub(u,v,q);w=ref_mod_mul(w,ws,q);}}}
}

static uint32_t xrand(uint64_t &st, uint32_t q) {
    st^=st<<13;st^=st>>7;st^=st<<17;return(uint32_t)(st%q);
}

/*---- Test functions ----*/

static int test_ref(uint32_t N, uint32_t q, uint32_t BATCH,
    const std::vector<uint32_t>&pp, const std::vector<uint32_t>&tw) {
    uint32_t logN=0; for(uint32_t t=N;t>1;t>>=1) logN++;
    std::cout<<"  test_ref(N="<<N<<",batch="<<BATCH<<",q="<<q<<")..."<<std::flush;
    uint64_t rng=42;
    std::vector<uint32_t> x(BATCH*N),gold(BATCH*N);
    for(uint32_t i=0;i<BATCH*N;i++){x[i]=xrand(rng,q);gold[i]=x[i];}
    for(uint32_t b=0;b<BATCH;b++){
        std::vector<uint32_t> row(gold.begin()+b*N,gold.begin()+(b+1)*N);
        ref_ntt(row,pp,q);
        for(uint32_t i=0;i<N;i++) gold[b*N+i]=row[i];
    }
    // Fresh copies (kernel modifies psi_powers for four-step)
    // psi_powers must be large enough: max(N, batch*N) for scratch
    std::vector<uint32_t> pc(pp);
    // For four-step, psi_powers buffer needs to be batch*N for scratch
    if (N > TILE_N) pc.resize(BATCH * N);
    std::vector<uint32_t> tc(tw);
    ntt_kernel(x.data(), pc.data(), tc.data(), q, BATCH, N, logN);
    int err=0;
    for(uint32_t i=0;i<BATCH*N;i++){
        if(x[i]!=gold[i]){if(err<5)std::cout<<"\n    MISMATCH["<<i<<"] exp="<<gold[i]<<" got="<<x[i];err++;}
    }
    std::cout<<(err==0?" PASS":"\n    FAIL("+std::to_string(err)+")")<<std::endl;
    return err;
}

static int test_lin(uint32_t N, uint32_t q,
    const std::vector<uint32_t>&pp, const std::vector<uint32_t>&tw) {
    uint32_t logN=0; for(uint32_t t=N;t>1;t>>=1) logN++;
    std::cout<<"  test_lin(N="<<N<<",q="<<q<<")..."<<std::flush;
    uint64_t rng=42;
    std::vector<uint32_t> av(N),bv(N),ab(N);
    for(uint32_t i=0;i<N;i++) av[i]=xrand(rng,q);
    for(uint32_t i=0;i<N;i++) bv[i]=xrand(rng,q);
    for(uint32_t i=0;i<N;i++) ab[i]=ref_mod_add(av[i],bv[i],q);

    size_t psz = (N > TILE_N) ? N : pp.size();

    std::vector<uint32_t> left(ab);
    {std::vector<uint32_t> p(pp); p.resize(psz); std::vector<uint32_t> t(tw);
     ntt_kernel(left.data(),p.data(),t.data(),q,1,N,logN);}
    std::vector<uint32_t> na(av);
    {std::vector<uint32_t> p(pp); p.resize(psz); std::vector<uint32_t> t(tw);
     ntt_kernel(na.data(),p.data(),t.data(),q,1,N,logN);}
    std::vector<uint32_t> nb(bv);
    {std::vector<uint32_t> p(pp); p.resize(psz); std::vector<uint32_t> t(tw);
     ntt_kernel(nb.data(),p.data(),t.data(),q,1,N,logN);}

    int err=0;
    for(uint32_t i=0;i<N;i++){uint32_t r=ref_mod_add(na[i],nb[i],q);if(left[i]!=r)err++;}
    std::cout<<(err==0?" PASS":"\n    FAIL("+std::to_string(err)+")")<<std::endl;
    return err;
}

static int test_range(uint32_t N, uint32_t q, uint32_t BATCH,
    const std::vector<uint32_t>&pp, const std::vector<uint32_t>&tw) {
    uint32_t logN=0; for(uint32_t t=N;t>1;t>>=1) logN++;
    std::cout<<"  test_range(N="<<N<<",batch="<<BATCH<<",q="<<q<<")..."<<std::flush;
    uint64_t rng=42;
    std::vector<uint32_t> x(BATCH*N);
    for(uint32_t i=0;i<BATCH*N;i++) x[i]=xrand(rng,q);
    std::vector<uint32_t> p(pp);
    if (N > TILE_N) p.resize(BATCH * N);
    std::vector<uint32_t> t(tw);
    ntt_kernel(x.data(),p.data(),t.data(),q,BATCH,N,logN);
    int err=0;
    for(uint32_t i=0;i<BATCH*N;i++) if(x[i]>=q)err++;
    std::cout<<(err==0?" PASS":"\n    FAIL("+std::to_string(err)+")")<<std::endl;
    return err;
}

int main() {
    int total=0;

    std::cout<<"=== Direct Path ===" << std::endl;
    uint32_t logns[]={8,9,10,11,12};
    for(auto ln:logns){
        uint32_t N=1u<<ln, q=generate_ntt_modulus(N,31), psi=find_psi(N,q);
        std::vector<uint32_t> pp,tw;
        make_direct_tables(N,q,psi,pp,tw);
        total+=test_ref(N,q,1,pp,tw);
        total+=test_ref(N,q,4,pp,tw);
        total+=test_lin(N,q,pp,tw);
        total+=test_range(N,q,4,pp,tw);
    }

    std::cout<<"\n=== Crypto Params ===" << std::endl;
    {uint32_t psi=find_psi(256,8380417);std::vector<uint32_t> pp,tw;make_direct_tables(256,8380417,psi,pp,tw);total+=test_ref(256,8380417,4,pp,tw);}
    {uint32_t psi=find_psi(512,12289);std::vector<uint32_t> pp,tw;make_direct_tables(512,12289,psi,pp,tw);total+=test_ref(512,12289,4,pp,tw);}
    {uint32_t psi=find_psi(1024,12289);std::vector<uint32_t> pp,tw;make_direct_tables(1024,12289,psi,pp,tw);total+=test_ref(1024,12289,2,pp,tw);}

    std::cout<<"\n=== Four-Step Path ===" << std::endl;
    uint32_t fs_logns[]={13,14,15,16};
    for(auto ln:fs_logns){
        uint32_t N=1u<<ln, q=generate_ntt_modulus(N,31), psi=find_psi(N,q);
        std::vector<uint32_t> pp,tw;
        make_four_step_tables(N,q,psi,pp,tw);
        total+=test_ref(N,q,1,pp,tw);
        if (ln <= 14) {
            total+=test_ref(N,q,2,pp,tw);
        }
        total+=test_lin(N,q,pp,tw);
        total+=test_range(N,q,1,pp,tw);
    }

    std::cout<<"\n==============================\n";
    std::cout<<(total==0?"ALL TESTS PASSED":"FAILURES DETECTED")<<std::endl;
    std::cout<<"=============================="<<std::endl;
    return total?1:0;
}