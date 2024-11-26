#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../parameter/parameter_g.cuh"

__device__ __forceinline__ uint32_t uadd_cc(uint32_t a, uint32_t b) {
	uint32_t r;
  
	asm volatile ("add.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ uint32_t uaddc_cc(uint32_t a, uint32_t b) {
	uint32_t r;
	
	asm volatile ("addc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ uint32_t uaddc(uint32_t a, uint32_t b) {
	uint32_t r;
	
	asm volatile ("addc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ uint32_t usub_cc(uint32_t a, uint32_t b) {
	uint32_t r;
  
	asm volatile ("sub.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ uint32_t usubc_cc(uint32_t a, uint32_t b) {
	uint32_t r;
	
	asm volatile ("subc.cc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ uint32_t usubc(uint32_t a, uint32_t b) {
	uint32_t r;
	
	asm volatile ("subc.u32 %0,%1,%2;" : "=r"(r) : "r"(a), "r"(b));
	return r;
}
  
__device__ __forceinline__ bool getCarry() {
	return uaddc(0, 0)!=0;
}
  
__device__ __forceinline__ uint64_t madwide(uint32_t a, uint32_t b, uint64_t c) {
	uint64_t r;
	
	asm volatile ("mad.wide.u32 %0,%1,%2,%3;" : "=l"(r) : "r"(a), "r"(b), "l"(c));
	return r;
}
  
__device__ __forceinline__ uint64_t madwidec_cc(uint32_t a, uint32_t b, uint64_t c) {
	uint64_t r;
	
	asm volatile ("{\n\t"
				  ".reg .u32 lo,hi;\n\t"
				  "mov.b64        {lo,hi},%3;\n\t"
				  "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
				  "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
				  "mov.b64        %0,{lo,hi};\n\t"
				  "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));                
	return r;
}
  
__device__ __forceinline__ uint64_t madwidec(uint32_t a, uint32_t b, uint64_t c) {
	uint64_t r;
	
	asm volatile ("{\n\t"
				  ".reg .u32 lo,hi;\n\t"
				  "mov.b64        {lo,hi},%3;\n\t"
				  "madc.lo.cc.u32 lo,%1,%2,lo;\n\t"
				  "madc.hi.cc.u32 hi,%1,%2,hi;\n\t"
				  "mov.b64        %0,{lo,hi};\n\t"
				  "}" : "=l"(r) : "r"(a), "r"(b), "l"(c));                
	return r;
}
  
__device__ __forceinline__ uint2 u2madwidec_cc(uint32_t a, uint32_t b, uint2 c) {
	uint2 r;
	
	asm volatile ("madc.lo.cc.u32  %0,%2,%3,%4;\n\t"
				  "madc.hi.cc.u32 %1,%2,%3,%5;" : "=r"(r.x), "=r"(r.y) : "r"(a), "r"(b), "r"(c.x), "r"(c.y));
	return r;
}
  
__device__ __forceinline__ uint32_t ulow(uint2 xy) {
	return xy.x;
}
  
__device__ __forceinline__ uint32_t uhigh(uint2 xy) {
	return xy.y;
}
  
__device__ __forceinline__ uint32_t ulow(uint64_t wide) {
	uint32_t r;
  
	asm volatile ("mov.b64 {%0,_},%1;" : "=r"(r) : "l"(wide));
	return r;
}
  
__device__ __forceinline__ uint32_t uhigh(uint64_t wide) {
	uint32_t r;
  
	asm volatile ("mov.b64 {_,%0},%1;" : "=r"(r) : "l"(wide));
	return r;
}
  
__device__ __forceinline__ uint64_t make_wide(uint32_t lo, uint32_t hi) {
	uint64_t r;
	
	asm volatile ("mov.b64 %0,{%1,%2};" : "=l"(r) : "r"(lo), "r"(hi));
	return r;
}
  
__device__ __forceinline__ uint64_t make_wide(uint2 xy) {
	return make_wide(xy.x, xy.y);
}


class chain_t {
  public:
  bool firstOperation;
  
  __device__ __forceinline__ chain_t() {
    firstOperation=true;
  }
  
  __device__ __forceinline__ chain_t(bool carry) {
    firstOperation=false;
    uadd_cc(carry ? 1 : 0, 0xFFFFFFFF);
  }
  
  __device__ __forceinline__ void reset() {
    firstOperation=true;
    uadd_cc(0, 0);
  }
  
  __device__ __forceinline__ void reset(bool carry) {
    firstOperation=false;
    uadd_cc(carry ? 1 : 0, 0xFFFFFFFF);
  }
  
  __device__ __forceinline__ bool getCarry() {
    return uaddc(0, 0)!=0;
  }
  
  __device__ __forceinline__ uint32_t add(uint32_t a, uint32_t b) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;
    return uaddc_cc(a, b);
  }
  
  __device__ __forceinline__ uint32_t sub(uint32_t a, uint32_t b) {
    if(firstOperation)
      uadd_cc(1, 0xFFFFFFFF);
    firstOperation=false;
    return usubc_cc(a, b);
  }
  
  __device__ __forceinline__ uint2 madwide(uint32_t a, uint32_t b, uint2 c) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;    
    return u2madwidec_cc(a, b, c);
  }

  __device__ __forceinline__ uint64_t madwide(uint32_t a, uint32_t b, uint64_t c) {
    if(firstOperation) 
      uadd_cc(0, 0);
    firstOperation=false;    
    return madwidec_cc(a, b, c);
  }
};

__device__ __forceinline__ static uint32_t qTerm(uint32_t lowWord, uint32_t np0) {
  uint64_t product = (uint64_t)lowWord * np0;
  uint32_t result = (uint32_t)(product & 0xFFFFFFFF); 
  return result;
}

__device__ __forceinline__ uint32_t computeNP0(uint32_t x) {
  uint32_t inv=x;

  inv=inv*(inv*x+14);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  inv=inv*(inv*x+2);
  return inv;
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_comp_gt(const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  // a>b --> b-a is negative
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    chain.sub(b[i], a[i]);
  return !chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_comp_ge(const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    chain.sub(a[i], b[i]);
  return chain.getCarry();
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_add(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.add(a[i], b[i]);
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_sub(uint32_t* r, const uint32_t* a, const uint32_t* b) {
  chain_t chain;
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i++)
    r[i]=chain.sub(a[i], b[i]);
}

template<uint32_t limbs>
__device__ __forceinline__ void mp_merge_cl(uint32_t* r, const uint64_t* evenOdd, bool carry) {
  chain_t chain(carry);
 
  r[0]=ulow(evenOdd[0]);
  for(int i=0;i<limbs/2-1;i++) {
    r[2*i+1]=chain.add(uhigh(evenOdd[i]), ulow(evenOdd[limbs/2 + i]));
    r[2*i+2]=chain.add(ulow(evenOdd[i+1]), uhigh(evenOdd[limbs/2 + i]));
  }
  r[limbs-1]=chain.add(uhigh(evenOdd[limbs/2-1]), 0);
}

template<uint32_t limbs>
__device__ __forceinline__ bool mp_mul_red_cl(uint64_t* evenOdd, const uint32_t* a, const uint32_t* b, const uint32_t* n) {
  uint64_t* even=evenOdd;
  uint64_t* odd=evenOdd + limbs/2;
  chain_t   chain;
  bool      carry=false;
  uint32_t  lo=0, q, c1, c2;
  
  // This routine can be used when max(a, b)+n < R (i.e. it doesn't carry out).  Hence the name cl for carryless.
  // Only works with an even number of limbs.
     
  #pragma unroll
  for(int32_t i=0;i<limbs/2;i++) {
    even[i]=make_wide(0, 0);
    odd[i]=make_wide(0, 0);
  }
  
  #pragma unroll
  for(int32_t i=0;i<limbs;i+=2) {
    if(i!=0) {
      // integrate lo
      chain.reset(carry);
      lo=chain.add(lo, ulow(even[0]));
      carry=chain.add(0, 0)!=0;
      even[0]=make_wide(lo, uhigh(even[0]));
    }

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(a[i], b[j], even[j/2]);
    c1=chain.add(0, 0);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(a[i], b[j+1], odd[j/2]);

    q=qTerm(ulow(even[0]),np0);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(q, n[j+1], odd[j/2]);

    chain.reset();
    even[0]=chain.madwide(q, n[0], even[0]);
    lo=uhigh(even[0]);
    #pragma unroll
    for(int j=2;j<limbs;j+=2)
      even[j/2-1]=chain.madwide(q, n[j], even[j/2]);
    c1=chain.add(c1, 0);
      
    // integrate lo
    
    chain.reset(carry);
    lo=chain.add(lo, ulow(odd[0]));
    carry=chain.add(0, 0)!=0;
    odd[0]=make_wide(lo, uhigh(odd[0]));

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      odd[j/2]=chain.madwide(a[i+1], b[j], odd[j/2]);
    c2=chain.add(0, 0);

    q=qTerm(ulow(odd[0]),np0);

    // shift odd by 64 bits

    chain.reset();
    odd[0]=chain.madwide(q, n[0], odd[0]);
    lo=uhigh(odd[0]);
    #pragma unroll
    for(int j=2;j<limbs;j+=2)
      odd[j/2-1]=chain.madwide(q, n[j], odd[j/2]);
    c2=chain.add(c2, 0);

    odd[limbs/2-1]=make_wide(0, 0);
    even[limbs/2-1]=make_wide(c1, c2);
    
    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(a[i+1], b[j+1], even[j/2]);

    chain.reset();
    #pragma unroll
    for(int j=0;j<limbs;j+=2)
      even[j/2]=chain.madwide(q, n[j+1], even[j/2]);
  }

  chain.reset(carry);
  lo=chain.add(lo, ulow(even[0]));
  carry=chain.add(0, 0)!=0;
  even[0]=make_wide(lo, uhigh(even[0]));
  return carry;
}