#include <iostream>
#include <gmp.h>
#include "fr.cuh"
#include "../parameter/parameter.cuh"
using namespace std;

void h_SUB(uint32_t* self, const uint32_t* rhs, uint32_t* result) {
    mpz_t a, b, c, mod;

    mpz_init(a);
    mpz_import(a, h_NUM, -1, sizeof(uint32_t), 0, 0, self);
    mpz_init(b);
    mpz_import(b, h_NUM, -1, sizeof(uint32_t), 0, 0, rhs);
    mpz_init(c);
    mpz_init(mod);
    mpz_import(mod, h_NUM, -1, sizeof(uint32_t), 0, 0, h_MODULUS);

    mpz_sub(c,a,b);
	mpz_mod(c,c,mod);

    size_t mark;
	mpz_export(result, &mark, -1, sizeof(uint32_t), 0, 0, c);
	while (mark<h_NUM)
		result[mark++] = 0;
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(mod);
}


void h_ADD(uint32_t* self, const uint32_t* rhs, uint32_t* result) {
    mpz_t a, b, c, mod;

    mpz_init(a);
    mpz_import(a, h_NUM, -1, sizeof(uint32_t), 0, 0, self);
    mpz_init(b);
    mpz_import(b, h_NUM, -1, sizeof(uint32_t), 0, 0, rhs);
    mpz_init(c);
    mpz_init(mod);
    mpz_import(mod, h_NUM, -1, sizeof(uint32_t), 0, 0, h_MODULUS);

    mpz_add(c,a,b);
	mpz_mod(c,c,mod);

    size_t mark;
	mpz_export(result, &mark, -1, sizeof(uint32_t), 0, 0, c);
	while (mark<h_NUM)
		result[mark++] = 0;
    
    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(mod);
}

void h_MUL(uint32_t* self, const uint32_t* rhs, uint32_t* result){
    mpz_t a, b, c, mod;

    mpz_init(a);
    mpz_import(a, h_NUM, -1, sizeof(uint32_t), 0, 0, self);
    mpz_init(b);
    mpz_import(b, h_NUM, -1, sizeof(uint32_t), 0, 0, rhs);
    mpz_init(c);
    mpz_init(mod);
    mpz_import(mod, h_NUM, -1, sizeof(uint32_t), 0, 0, h_MODULUS);

    mpz_mul(c,a,b);
	mpz_mod(c,c,mod);

    size_t mark;
	mpz_export(result, &mark, -1, sizeof(uint32_t), 0, 0, c);
    	while (mark<h_NUM)
		result[mark++] = 0;

    mpz_clear(a);
    mpz_clear(b);
    mpz_clear(c);
    mpz_clear(mod);
}

void to_mont(uint32_t* data){
    uint32_t temp[h_NUM+1]={};
	temp[h_NUM]=0x00000001;

    mpz_t mod;
    mpz_init(mod);
    mpz_import(mod, h_NUM, -1, sizeof(uint32_t), 0, 0, h_MODULUS);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, h_NUM, -1, sizeof(uint32_t), 0, 0, data);

    mpz_t mpz_temp;
    mpz_init(mpz_temp);
    mpz_import(mpz_temp, h_NUM+1, -1, sizeof(uint32_t), 0, 0, temp);
    
    mpz_mul(a,a,mpz_temp);
	mpz_mod(a,a,mod);
    size_t mark;
	mpz_export(data, &mark, -1, sizeof(uint32_t), 0, 0, a);
	while (mark<h_NUM)
		data[mark++] = 0;

    mpz_clear(mod);
    mpz_clear(a);
    mpz_clear(mpz_temp);
}

void mont_back(uint32_t* data){
	uint32_t temp[h_NUM+1]={};
	temp[h_NUM]=0x00000001;

    mpz_t mod;
    mpz_init(mod);
    mpz_import(mod, h_NUM, -1, sizeof(uint32_t), 0, 0, h_MODULUS);

    mpz_t a;
    mpz_init(a);
    mpz_import(a, h_NUM, -1, sizeof(uint32_t), 0, 0, data);

	mpz_t mpz_temp;
	mpz_init(mpz_temp);
	mpz_import(mpz_temp, h_NUM+1, -1, sizeof(uint32_t), 0, 0, temp);
	
	mpz_invert(mpz_temp,mpz_temp,mod);
	mpz_mul(a,a,mpz_temp);
	mpz_mod(a,a,mod);
	size_t mark;
	mpz_export(data, &mark, -1, sizeof(uint32_t), 0, 0, a);
	while (mark<h_NUM)
		data[mark++] = 0;

    mpz_clear(mod);
    mpz_clear(a);
    mpz_clear(mpz_temp);
}