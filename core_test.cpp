#include "secp_core.cuh"
#include <boost/multiprecision/cpp_int.hpp>
#include <random>
#include <cstdio>
#include <cstdlib>
using boost::multiprecision::cpp_int;
static const cpp_int p=(cpp_int(1)<<256)-(cpp_int(1)<<32)-977;
cpp_int as_int(const FE &a){cpp_int x=0;for(int i=7;i>=0;i--){x<<=32;x+=a.v[i];}return x;}
FE from_int(cpp_int x){FE a;for(int i=0;i<8;i++){a.v[i]=(uint32_t)(x&0xffffffff);x>>=32;}return a;}
FE affine_x(const PointJ &q){return fe_mul(q.X,fe_inv(fe_sqr(q.Z)));}
U96 make_u96(unsigned __int128 n){return {(uint64_t)n,(uint32_t)(n>>64)};}
int main(){std::mt19937_64 rng(123456);
  for(int i=0;i<200000;i++){
    cpp_int a=0,b=0;for(int j=0;j<4;j++){a<<=64;a+=rng();b<<=64;b+=rng();}a%=p;b%=p;
    FE A=from_int(a),B=from_int(b);
    if(as_int(fe_mul(A,B))!=(a*b)%p){printf("field mul failed at %d\n",i);return 1;}
    if(as_int(fe_add(A,B))!=(a+b)%p){printf("field add failed at %d\n",i);return 1;}
    if(as_int(fe_sub(A,B))!=(a-b+p)%p){printf("field sub failed at %d\n",i);return 1;}
  }
  puts("PASS: 200000 field multiply/add/sub comparisons");
  for(int i=1;i<=100;i++){
    FE a; for(int j=0;j<8;j++)a.v[j]=(uint32_t)rng();
    if(fe_iszero(a))a=fe_one();
    if(!fe_equal(fe_mul(a,fe_inv(a)),fe_one())){printf("inverse failed %d\n",i);return 1;}
  }
  puts("PASS: 100 field inversion checks");
  // Check scalar multiplication against a separate affine cpp_int implementation.
  auto mod=[](cpp_int x){x%=p;if(x<0)x+=p;return x;};
  cpp_int gx("0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
  cpp_int gy("0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
  auto refmul=[&](unsigned long long k){cpp_int rx=0,ry=0;bool inf=true;cpp_int qx=gx,qy=gy;while(k){if(k&1){if(inf){rx=qx;ry=qy;inf=false;}else{cpp_int lam=mod((qy-ry)*boost::multiprecision::powm(mod(qx-rx),p-2,p));cpp_int nx=mod(lam*lam-rx-qx),ny=mod(lam*(rx-nx)-ry);rx=nx;ry=ny;}}cpp_int lam=mod(3*qx*qx*boost::multiprecision::powm(mod(2*qy),p-2,p));cpp_int nx=mod(lam*lam-2*qx),ny=mod(lam*(qx-nx)-qy);qx=nx;qy=ny;k>>=1;}return inf?cpp_int(0):rx;};
  for(unsigned long long k=1;k<100;k++){PointJ q=point_mul_g({k,0});if(as_int(affine_x(q))!=refmul(k)){printf("point mul failed %llu\n",k);return 1;}}
  puts("PASS: scalar multiplication 1..99");
  // General Jacobian addition/subtraction from P(k+S)-S*G = P(k).
  for(int i=0;i<200;i++){
    unsigned long long k=(rng()%1000000)+1, h=(rng()%100000)+1;
    PointJ pk=point_mul_g({k,0}), ph=point_mul_g({h,0});
    PointJ pm=point_mul_g({k>=h?k-h:h-k,0}); ph.Y=fe_neg(ph.Y);
    PointJ got=point_add(pk,ph);
    if(!fe_equal(affine_x(got),affine_x(pm))){printf("Jacobian subtract failed %d\n",i);return 1;}
  }
  puts("PASS: 200 Jacobian subtraction comparisons");
  // Projective differential recurrence, with independently computed points.
  for(int i=0;i<100;i++){
    unsigned h=(rng()%10000)+1, k=(rng()%100000)+h+1;
    PointJ ph=point_mul_g({h,0}), pk=point_mul_g({k,0}), pm=point_mul_g({k-h,0}), pn=point_mul_g({k+h,0});
    FE nx,nz; xonly_projective_step(pm.X,fe_sqr(pm.Z),pk.X,fe_sqr(pk.Z),ph.X,fe_sqr(ph.Z),&nx,&nz);
    FE got=fe_mul(nx,fe_inv(nz));
    if(!fe_equal(got,affine_x(pn))){printf("projective recurrence failed %d\n",i);return 1;}
  }
  puts("PASS: 100 projective X-only recurrence comparisons");
  // Stress the affine batched-inversion recurrence with random 80-bit scalar pairs.
  for(int i=0;i<400;i++){
    unsigned __int128 h=((unsigned __int128)(rng()&0xffff)<<64)|rng();
    unsigned __int128 k=((unsigned __int128)(rng()&0xffff)<<64)|rng();
    h=(h%(((unsigned __int128)1<<79)-1))+1;
    k=(k%(((unsigned __int128)1<<79)-1))+h+1;
    PointJ ph=point_mul_g(make_u96(h)),pk=point_mul_g(make_u96(k));
    PointJ pm=point_mul_g(make_u96(k-h)),pn=point_mul_g(make_u96(k+h));
    FE a=affine_x(ph),x=affine_x(pk),prev=affine_x(pm),d=fe_sub(x,a);
    FE id=fe_inv(d), n=fe_add(fe_mul(fe_mul(a,x),fe_add(x,a)),from_int(14));
    FE xn=fe_sub(fe_double(fe_mul(n,fe_sqr(id))),prev);
    if(!fe_equal(xn,affine_x(pn))){printf("random 80-bit affine recurrence failed %d\n",i);return 1;}
  }
  puts("PASS: 400 randomized 80-bit affine recurrence comparisons");
  // Verify warp batch inversion formula (prefix/suffix products).
  for(int batch=0;batch<40;batch++){
    FE z[32],pref[32]; FE product=fe_one();
    for(int i=0;i<32;i++){for(int j=0;j<8;j++)z[i].v[j]=(uint32_t)rng();if(fe_iszero(z[i]))z[i]=fe_one();product=fe_mul(product,z[i]);pref[i]=product;}
    FE invProduct=fe_inv(product),suffix=fe_one();
    for(int i=31;i>=0;i--){FE before=(i==0)?fe_one():pref[i-1];FE invz=fe_mul(invProduct,fe_mul(before,suffix));if(!fe_equal(fe_mul(invz,z[i]),fe_one())){printf("batch inverse failed %d %d\n",batch,i);return 1;}suffix=fe_mul(suffix,z[i]);}
  }
  puts("PASS: 1280 batch-inversion identities");
  for(int batch=0;batch<100;batch++){
    FE z[32],prefix[32],suffix[32],old[32];
    for(int i=0;i<32;i++){for(int j=0;j<8;j++)z[i].v[j]=(uint32_t)rng();if(fe_iszero(z[i]))z[i]=fe_one();prefix[i]=z[i];suffix[i]=z[i];}
    for(int delta=1;delta<32;delta<<=1){for(int i=0;i<32;i++)old[i]=prefix[i];for(int i=delta;i<32;i++)prefix[i]=fe_mul(old[i],old[i-delta]);}
    for(int delta=1;delta<32;delta<<=1){for(int i=0;i<32;i++)old[i]=suffix[i];for(int i=0;i+delta<32;i++)suffix[i]=fe_mul(old[i],old[i+delta]);}
    FE invTotal=fe_inv(prefix[31]);
    for(int i=0;i<32;i++){
      FE before=i==0?fe_one():prefix[i-1], after=i==31?fe_one():suffix[i+1];
      FE invz=fe_mul(invTotal,fe_mul(before,after));
      if(!fe_equal(fe_mul(invz,z[i]),fe_one())){printf("warp scan inversion failed batch=%d lane=%d\n",batch,i);return 1;}
    }
  }
  puts("PASS: 3200 warp-shuffle scan inversion checks");
  // 80-bit prefix-distance pair arithmetic checks.
  for(int i=0;i<100000;i++){
    unsigned __int128 pfx=((unsigned __int128)(rng()&0xffff)<<64)|rng();
    unsigned __int128 tgt=((unsigned __int128)(rng()&0xffff)<<64)|rng();
    pfx&=(((unsigned __int128)1<<80)-1);tgt&=(((unsigned __int128)1<<80)-1);
    uint32_t ph=(uint32_t)(pfx>>48),th=(uint32_t)(tgt>>48);
    uint64_t pl=(uint64_t)(pfx&((((unsigned __int128)1)<<48)-1)),tl=(uint64_t)(tgt&((((unsigned __int128)1)<<48)-1));
    uint32_t dh;uint64_t dl;
    if(ph>th||(ph==th&&pl>=tl)){uint64_t b=pl<tl;dl=(pl-tl)&0x0000FFFFFFFFFFFFULL;dh=ph-th-(uint32_t)b;}
    else{uint64_t b=tl<pl;dl=(tl-pl)&0x0000FFFFFFFFFFFFULL;dh=th-ph-(uint32_t)b;}
    unsigned __int128 got=((unsigned __int128)dh<<48)|dl;
    unsigned __int128 want=pfx>tgt?pfx-tgt:tgt-pfx;
    if(got!=want){puts("prefix-distance test failed");return 1;}
  }
  puts("PASS: 100000 exact 80-bit prefix-distance checks");

  // Full 10,000-scalar residue-lane scan, matching the CUDA kernel design.
  {
    const unsigned S=256, U=10000;

    PointJ H=point_mul_g({S,0}); FE hx=affine_x(H), fourteen=fe_zero(); fourteen.v[0]=14;
    struct Lane { FE prev,cur; unsigned kcur; bool hasCur; } lanes[S];
    struct Cand { bool ok=false; unsigned __int128 d=0; unsigned k=0; FE x; } best;
    auto consider=[&](FE x,unsigned k){
      unsigned __int128 prefix=((unsigned __int128)x.v[7]<<48)|((unsigned __int128)x.v[6]<<16)|(x.v[5]>>16);
      unsigned __int128 tar=((unsigned __int128)3<<72);
      unsigned __int128 diff=prefix>tar?prefix-tar:tar-prefix;
      unsigned __int128 d=diff;
      if(!best.ok||d<best.d||(d==best.d&&k<best.k)){best.ok=true;best.d=d;best.k=k;best.x=x;}
    };
    for(unsigned t=0;t<S;t++){
      unsigned k0=1+t,k1=k0+S;
      PointJ pc=point_mul_g({k1,0}), nh=H; nh.Y=fe_neg(nh.Y);
      PointJ pp=point_add(pc,nh);
      lanes[t].prev=affine_x(pp);lanes[t].cur=affine_x(pc);lanes[t].kcur=k1;lanes[t].hasCur=(k1<=U);
      if(k0<=U)consider(lanes[t].prev,k0);
      if(k1<=U)consider(lanes[t].cur,k1);
    }
    unsigned maxIndex=(U-1)/S;
    for(unsigned round=0;round<(maxIndex>1?maxIndex-1:0);round++){
      FE d[S],pref[S];FE prod=fe_one();bool active[S];
      for(unsigned t=0;t<S;t++){
        unsigned nk=lanes[t].kcur+S;active[t]=lanes[t].hasCur&&nk<=U;
        d[t]=active[t]?fe_sub(lanes[t].cur,hx):fe_one();prod=fe_mul(prod,d[t]);pref[t]=prod;
      }
      FE invprod=fe_inv(prod),suffix=fe_one(), invd[S];
      for(int t=(int)S-1;t>=0;t--){FE before=t==0?fe_one():pref[t-1];invd[t]=fe_mul(invprod,fe_mul(before,suffix));suffix=fe_mul(suffix,d[t]);}
      for(unsigned t=0;t<S;t++)if(active[t]){
        FE id2=fe_sqr(invd[t]);
        FE n=fe_add(fe_mul(fe_mul(hx,lanes[t].cur),fe_add(lanes[t].cur,hx)),fourteen);
        FE nx=fe_sub(fe_double(fe_mul(n,id2)),lanes[t].prev);
        lanes[t].prev=lanes[t].cur;lanes[t].cur=nx;lanes[t].kcur+=S;consider(nx,lanes[t].kcur);
      }
    }
    if(!best.ok||best.k!=9325||as_int(best.x)!=cpp_int("0x02fd360f1680b7b65f49f64d5a648e8c97e4a756fe88e10791fce9b6295cfc85")){
      printf("complete lane scan mismatch: scalar=%u\n",best.k);return 1;
    }
    puts("PASS: complete 1..10000 affine batch-inversion scan; best scalar 9325");
  }
  puts("ALL CORE TESTS PASS");
}
