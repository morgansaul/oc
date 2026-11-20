#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

struct FE { uint32_t v[8]; };
struct U96 { uint64_t lo; uint32_t hi; };
struct PointJ { FE X, Y, Z; };

HD FE fe_zero() { FE r = {}; return r; }
HD FE fe_one()  { FE r = {}; r.v[0] = 1; return r; }
HD bool fe_iszero(const FE &a) {
    uint32_t x = 0; for (int i=0;i<8;i++) x |= a.v[i]; return x == 0;
}
HD bool fe_equal(const FE &a, const FE &b) {
    uint32_t x = 0; for (int i=0;i<8;i++) x |= (a.v[i] ^ b.v[i]); return x == 0;
}
HD bool fe_ge_p(const FE &a) {
    const uint32_t p[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    for (int i=7;i>=0;i--) { if (a.v[i] > p[i]) return true; if (a.v[i] < p[i]) return false; }
    return true;
}
HD FE fe_add(const FE &a, const FE &b) {
    const uint32_t p[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    FE r; uint64_t c=0;
    for (int i=0;i<8;i++) { uint64_t z=(uint64_t)a.v[i]+b.v[i]+c; r.v[i]=(uint32_t)z; c=z>>32; }
    if (c || fe_ge_p(r)) {
        uint64_t borrow=0;
        for (int i=0;i<8;i++) { uint64_t x=r.v[i], y=(uint64_t)p[i]+borrow; r.v[i]=(uint32_t)(x-y); borrow=(x<y); }
    }
    return r;
}
HD FE fe_sub(const FE &a, const FE &b) {
    const uint32_t p[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    FE r; uint64_t borrow=0;
    for (int i=0;i<8;i++) { uint64_t x=a.v[i], y=(uint64_t)b.v[i]+borrow; r.v[i]=(uint32_t)(x-y); borrow=(x<y); }
    if (borrow) { uint64_t c=0; for (int i=0;i<8;i++) { uint64_t z=(uint64_t)r.v[i]+p[i]+c; r.v[i]=(uint32_t)z; c=z>>32; } }
    return r;
}
HD FE fe_neg(const FE &a) { return fe_iszero(a) ? a : fe_sub(fe_zero(), a); }

HD void fe_add_u64_to_words(uint32_t *t, int idx, uint64_t value) {
    uint64_t cur=(uint64_t)t[idx]+(uint32_t)value;
    t[idx]=(uint32_t)cur;
    uint64_t carry=(value>>32)+(cur>>32);
    idx++;
    while (carry) { cur=(uint64_t)t[idx]+carry; t[idx]=(uint32_t)cur; carry=cur>>32; idx++; }
}

// Exact 256x256 -> 512 multiplication followed by reduction using
// 2^256 == 2^32 + 977 (mod p).
HD FE fe_mul(const FE &a, const FE &b) {
    uint32_t t[24] = {};
    for (int i=0;i<8;i++) {
        uint64_t carry=0;
        for (int j=0;j<8;j++) {
            uint64_t z=(uint64_t)a.v[i]*(uint64_t)b.v[j] + (uint64_t)t[i+j] + carry;
            t[i+j]=(uint32_t)z; carry=z>>32;
        }
        int k=i+8;
        while (carry) { uint64_t z=(uint64_t)t[k]+carry; t[k]=(uint32_t)z; carry=z>>32; k++; }
    }
    for (int rounds=0; rounds<32; rounds++) {
        int top=-1;
        for (int i=23;i>=8;i--) { if (t[i]) { top=i; break; } }
        if (top<0) break;
        uint32_t w=t[top]; t[top]=0;
        fe_add_u64_to_words(t, top-8, (uint64_t)w*977u);
        fe_add_u64_to_words(t, top-7, (uint64_t)w);
    }
    FE r; for (int i=0;i<8;i++) r.v[i]=t[i];
    if (fe_ge_p(r)) {
        const uint32_t p[8] = {0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
        uint64_t borrow=0;
        for (int i=0;i<8;i++) { uint64_t x=r.v[i], y=(uint64_t)p[i]+borrow; r.v[i]=(uint32_t)(x-y); borrow=(x<y); }
    }
    return r;
}
HD FE fe_sqr(const FE &a) { return fe_mul(a,a); }
HD FE fe_double(const FE &a) { return fe_add(a,a); }
HD FE fe_mul_small(const FE &a, uint32_t c) {
    FE x=a, r=fe_zero();
    while (c) { if (c&1) r=fe_add(r,x); x=fe_double(x); c>>=1; }
    return r;
}
HD FE fe_inv(const FE &a) {
    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    const uint32_t e[8] = {0xFFFFFC2D,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
    FE r=fe_one();
    for (int i=255;i>=0;i--) { r=fe_sqr(r); if ((e[i>>5]>>(i&31))&1u) r=fe_mul(r,a); }
    return r;
}

HD U96 u96_add64(U96 a, uint64_t b) {
    U96 r; r.lo=a.lo+b; r.hi=a.hi+(r.lo<a.lo); return r;
}
HD U96 u96_sub64(U96 a, uint64_t b) {
    U96 r; r.lo=a.lo-b; r.hi=a.hi-(a.lo<b); return r;
}
HD bool u96_le(U96 a, U96 b) { return (a.hi<b.hi) || (a.hi==b.hi && a.lo<=b.lo); }
HD bool u96_lt(U96 a, U96 b) { return (a.hi<b.hi) || (a.hi==b.hi && a.lo<b.lo); }
HD bool u96_bit(U96 a, int bit) { return bit>=64 ? ((a.hi>>(bit-64))&1u)!=0 : ((a.lo>>bit)&1u)!=0; }

HD PointJ point_inf() { PointJ r; r.X=fe_zero(); r.Y=fe_one(); r.Z=fe_zero(); return r; }
HD bool point_isinf(const PointJ &p) { return fe_iszero(p.Z); }
HD PointJ point_double(const PointJ &p) {
    if (point_isinf(p) || fe_iszero(p.Y)) return point_inf();
    FE A=fe_sqr(p.X), B=fe_sqr(p.Y), C=fe_sqr(B);
    FE D=fe_double(fe_sub(fe_sub(fe_sqr(fe_add(p.X,B)),A),C));
    FE E=fe_add(fe_add(A,A),A), F=fe_sqr(E);
    FE X=fe_sub(F,fe_double(D));
    FE Y=fe_sub(fe_mul(E,fe_sub(D,X)),fe_mul_small(C,8));
    FE Z=fe_double(fe_mul(p.Y,p.Z));
    PointJ r={X,Y,Z}; return r;
}
HD PointJ point_add(const PointJ &p, const PointJ &q) {
    if (point_isinf(p)) return q;
    if (point_isinf(q)) return p;
    FE z1z1=fe_sqr(p.Z), z2z2=fe_sqr(q.Z);
    FE U1=fe_mul(p.X,z2z2), U2=fe_mul(q.X,z1z1);
    FE S1=fe_mul(fe_mul(p.Y,q.Z),z2z2), S2=fe_mul(fe_mul(q.Y,p.Z),z1z1);
    if (fe_equal(U1,U2)) {
        if (fe_equal(S1,S2)) return point_double(p);
        return point_inf();
    }
    FE H=fe_sub(U2,U1), I=fe_sqr(fe_double(H)), J=fe_mul(H,I);
    FE r=fe_double(fe_sub(S2,S1)), V=fe_mul(U1,I);
    FE X=fe_sub(fe_sub(fe_sqr(r),J),fe_double(V));
    FE Y=fe_sub(fe_mul(r,fe_sub(V,X)),fe_double(fe_mul(S1,J)));
    FE Z=fe_mul(fe_sub(fe_sub(fe_sqr(fe_add(p.Z,q.Z)),z1z1),z2z2),H);
    PointJ out={X,Y,Z}; return out;
}
HD PointJ point_add_mixed(const PointJ &p, const FE &qx, const FE &qy) {
    if (point_isinf(p)) { PointJ r={qx,qy,fe_one()}; return r; }
    FE z1z1=fe_sqr(p.Z), U2=fe_mul(qx,z1z1), S2=fe_mul(fe_mul(qy,p.Z),z1z1);
    FE H=fe_sub(U2,p.X), r0=fe_sub(S2,p.Y);
    if (fe_iszero(H)) { if (fe_iszero(r0)) return point_double(p); return point_inf(); }
    FE HH=fe_sqr(H), I=fe_mul_small(HH,4), J=fe_mul(H,I), r=fe_double(r0), V=fe_mul(p.X,I);
    FE X=fe_sub(fe_sub(fe_sqr(r),J),fe_double(V));
    FE Y=fe_sub(fe_mul(r,fe_sub(V,X)),fe_double(fe_mul(p.Y,J)));
    FE Z=fe_sub(fe_sub(fe_sqr(fe_add(p.Z,H)),z1z1),HH);
    PointJ out={X,Y,Z}; return out;
}
HD PointJ point_mul_g(U96 k) {
    FE gx={{0x16F81798,0x59F2815B,0x2DCE28D9,0x029BFCDB,0xCE870B07,0x55A06295,0xF9DCBBAC,0x79BE667E}};
    FE gy={{0xFB10D4B8,0x9C47D08F,0xA6855419,0xFD17B448,0x0E1108A8,0x5DA4FBFC,0x26A3C465,0x483ADA77}};
    PointJ r=point_inf();
    for (int i=95;i>=0;i--) { r=point_double(r); if (u96_bit(k,i)) r=point_add_mixed(r,gx,gy); }
    return r;
}

// State is x(P) in projective form X/Z, with X_J and Z_J^2 from Jacobian points.
HD void xonly_projective_step(const FE &prevX, const FE &prevZ,
                              const FE &curX, const FE &curZ,
                              const FE &hX, const FE &hZ,
                              FE *nextX, FE *nextZ) {
    FE d=fe_sub(fe_mul(hZ,curX),fe_mul(hX,curZ));
    FE d2=fe_sqr(d);
    FE n=fe_add(fe_add(fe_mul(fe_mul(hX,hZ),fe_sqr(curX)),
                       fe_mul(fe_sqr(hX),fe_mul(curX,curZ))),
                fe_mul_small(fe_mul(fe_sqr(hZ),fe_sqr(curZ)),14));
    *nextX=fe_sub(fe_double(fe_mul(prevZ,n)),fe_mul(prevX,d2));
    *nextZ=fe_mul(prevZ,d2);
}
