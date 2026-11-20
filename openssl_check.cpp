#include "secp_core.cuh"
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/bn.h>
#include <random>
#include <cstdio>
#include <cstdint>
#include <string>
#include <iostream>
static BIGNUM* bn128(unsigned __int128 x){ unsigned char b[16]; for(int i=15;i>=0;--i){b[i]=(unsigned char)x;x>>=8;} return BN_bin2bn(b,16,nullptr); }
static FE from_bn(const BIGNUM* b){ unsigned char out[32]; BN_bn2binpad(b,out,32); FE x{}; for(int i=0;i<8;i++){uint32_t w=0; int off=28-4*i; for(int j=0;j<4;j++) w=(w<<8)|out[off+j]; x.v[i]=w;} return x; }
static bool openssl_x(const EC_GROUP* g, BN_CTX* ctx, unsigned __int128 k, FE* x){ BIGNUM* n=bn128(k); EC_POINT* q=EC_POINT_new(g); BIGNUM* xx=BN_new();BIGNUM* yy=BN_new();bool ok=EC_POINT_mul(g,q,n,nullptr,nullptr,ctx)==1 && EC_POINT_get_affine_coordinates(g,q,xx,yy,ctx)==1; if(ok)*x=from_bn(xx); BN_free(n);EC_POINT_free(q);BN_free(xx);BN_free(yy);return ok; }
static FE affine_x(const PointJ& p){ return fe_mul(p.X,fe_inv(fe_sqr(p.Z))); }
int main(){
 EC_GROUP* g=EC_GROUP_new_by_curve_name(NID_secp256k1); BN_CTX* ctx=BN_CTX_new(); if(!g||!ctx)return 2;
 std::mt19937_64 rng(0x51ec2561);
 for(int i=0;i<120;i++){
   unsigned __int128 h=((unsigned __int128)(rng()&0xffff)<<64)|rng();
   unsigned __int128 k=((unsigned __int128)(rng()&0xffff)<<64)|rng();
   h=(h%(((unsigned __int128)1<<78)-1))+1;
   k=(k%(((unsigned __int128)1<<78)-1))+h+1;
   FE xh,xk,xkmh,xkp;
   if(!openssl_x(g,ctx,h,&xh)||!openssl_x(g,ctx,k,&xk)||!openssl_x(g,ctx,k-h,&xkmh)||!openssl_x(g,ctx,k+h,&xkp)){std::puts("OpenSSL point generation failure");return 2;}
   FE d=fe_sub(xk,xh); FE id=fe_inv(d); FE n=fe_add(fe_mul(fe_mul(xh,xk),fe_add(xk,xh)),[](){FE z=fe_zero();z.v[0]=14;return z;}());
   FE got=fe_sub(fe_double(fe_mul(n,fe_sqr(id))),xkmh);
   if(!fe_equal(got,xkp)){std::printf("recurrence mismatch at case %d\n",i);return 1;}
   // Also check the custom Jacobian scalar multiplication independently.
   FE ours=affine_x(point_mul_g({(uint64_t)k,(uint32_t)(k>>64)}));
   if(!fe_equal(ours,xk)){std::printf("custom scalar multiplication mismatch at case %d\n",i);return 1;}
 }
 std::puts("PASS: 120 randomized 80-bit recurrence and scalar-multiplication checks against OpenSSL secp256k1");
 EC_GROUP_free(g);BN_CTX_free(ctx);return 0;
}
