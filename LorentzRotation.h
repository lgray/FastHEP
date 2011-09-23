#include "LorentzVector.h"

#ifndef __fhep_LorentzRotation_h__
#define __fhep_LorentzRotation_h__

#include <vecLib/cblas.h>
#include <iostream>
#include <cmath>

#ifndef __CINT__
#include <smmintrin.h>
#endif

namespace fhep {

  class LorentzRotation {
    
  private:
    float m[16] __attribute__ ((aligned (16))); // column major
    
    /*
      tt tx ty tz
      xt xx xy xz
      yt yx yy yz
      zt zx zy zz
    */
    
  public:
    
    LorentzRotation() { // initialize to identity
      memset(m,0,16*sizeof(float));
      m[0] = m[4] = m[8] = m[12] = 1.0f;
    }
    
    LorentzRotation(const LorentzRotation& o) {
      memcpy(m,o.m,16*sizeof(float));
    }
    
    LorentzRotation(const float* o) {
      memcpy(m,o,16*sizeof(float));
    }
    
    LorentzRotation(float xx, float xy, float xz, float xt,
		    float yx, float yy, float yz, float yt,
		    float zx, float zy, float zz, float zt,
		    float tx, float ty, float tz, float tt) {
      float tmp[16] = {tt,xt,yt,zt,tx,xx,yx,zx,ty,xy,yy,zy,tz,xz,yz,zz};
      memcpy(m,tmp,16*sizeof(float));
    }
    
    LorentzRotation(float x, float y, float z) {
      setBoost(x,y,z);
    }
    
    void setBoost(const float& bx, 
		  const float& by, 
		  const float& bz) {
      if( bx == by == bz == 0.0f ) {    
	memset(m,0,16*sizeof(float));
	m[0] = m[4] = m[8] = m[12] = 1.0f;
      }
      __m128 b = _mm_setr_ps(1.0f,bx,by,bz);
      __m128 one = _mm_setr_ps(0.0f,1.0f,0.0f,0.0f);
      
      float gamma = 1.0f/std::sqrt(1.0f-_mm_dp_ps(b,b,0xe1)[0]); 
      // rsqrt is fast but inaccurate, and I think it is ok to waste a bit of time here
      // for the sake of accurate boost matrices
      float bgamma = gamma*gamma/(1.0f + gamma);
      
      __m128 t = _mm_setr_ps(gamma,-gamma,-gamma,-gamma);
      __m128 scal = _mm_setr_ps(-gamma,bgamma,bgamma,bgamma);
      
      // we can now construct the matrix in a few vector operations
      
      _mm_store_ps(&m[0],_mm_mul_ps(t,b));
      //        3  2  1  0
      // 0xff = 11 11 11 11 -> 3rd word of each input goes to correspond value in output
      // we want to move the 1+bgamma term down each iteration
      // so i == 1 means do nothing    11 10 01 00 // no shuffle
      //    i == 2 flip one and two    11 01 10 00 
      //    i == 3 flip two and three  10 11 01 00
      _mm_store_ps(&m[4],_mm_add_ps(one,_mm_mul_ps(scal,_mm_mul_ps(_mm_set1_ps(b[1]),b))));
      scal = _mm_shuffle_ps(scal,scal,0xd8); // 1 -> 2
      one = _mm_shuffle_ps(one,one,0xd8);
      _mm_store_ps(&m[8],_mm_add_ps(one,_mm_mul_ps(scal,_mm_mul_ps(_mm_set1_ps(b[2]),b))));
      scal = _mm_shuffle_ps(scal,scal,0xb4); // 2 -> 3
      one = _mm_shuffle_ps(one,one,0xb4);
      _mm_store_ps(&m[12],_mm_add_ps(one,_mm_mul_ps(scal,_mm_mul_ps(_mm_set1_ps(b[3]),b))));
    }
    
    inline LorentzRotation& operator= (const LorentzRotation& o) {
      memcpy(m,o.m,16*sizeof(float));
      return *this;
    }
    // I am not going to implement the "Transform" functions
    // they are redundant
    // here are operations on other matrices
    inline LorentzRotation operator* (const LorentzRotation& o) const {
      float res[16] __attribute__ ((aligned (16)));
      cblas_sgemm(CblasColMajor,
		  CblasNoTrans,
		  CblasNoTrans,
		  4,4,4,
		  1.0,m,4,
		  o.m,4,
		  0,res,4); // computes c = m*o for a 4x4 matrix of floats, putting the result into m
      return LorentzRotation(res);
    }
    inline LorentzRotation& operator*= (const LorentzRotation& o) {
      cblas_sgemm(CblasColMajor,
		  CblasNoTrans,
		  CblasNoTrans,
		  4,4,4,
		  1.0,m,4,
		  o.m,4,
		  0,m,4); // computes c = m*o for a 4x4 matrix of floats, putting the result into m
      return *this;
    }  
    // inversion
    LorentzRotation operator~() const {
      float res[16] __attribute__ ((aligned (16)));
      __m128 c0 = _mm_load_ps(&m[0]);
      __m128 c1 = _mm_load_ps(&m[4]);
      __m128 c2 = _mm_load_ps(&m[8]);
      __m128 c3 = _mm_load_ps(&m[12]);
      c0 = _mm_mul_ps(c0,_mm_setr_ps(1.0f,-1.0f,-1.0f,-1.0f));
      // tt xt yt, zy -> tt -xt -yt -zt for the starting column
      _MM_TRANSPOSE4_PS(c0,c1,c2,c3);
      // c0 now r0, etc
      // spatial parts are SO(3) matrix, so we're done there
      // other temporal bits (that were a row) just need to be multiplied by -1  
      c0 = _mm_mul_ps(c0,_mm_setr_ps(1.0f,-1.0f,-1.0f,-1.0f));
      _mm_store_ps(&res[0],c0);
      _mm_store_ps(&res[4],c1);
      _mm_store_ps(&res[8],c2);
      _mm_store_ps(&res[12],c3);
      return LorentzRotation(res);
    }
    inline LorentzRotation inverse() const { return operator~(); }
    
    void invert() {
      __m128 c0 = _mm_load_ps(&m[0]);
      __m128 c1 = _mm_load_ps(&m[4]);
      __m128 c2 = _mm_load_ps(&m[8]);
      __m128 c3 = _mm_load_ps(&m[12]);
      c0 = _mm_mul_ps(c0,_mm_setr_ps(1.0f,-1.0f,-1.0f,-1.0f));
      // tt xt yt, zy -> tt -xt -yt -zt for the starting column
      _MM_TRANSPOSE4_PS(c0,c1,c2,c3);
      // c0 now r0, etc
      // spatial parts are SO(3) matrix, so we're done there
      // other temporal bits (that were a row) just need to be multiplied by -1  
      c0 = _mm_mul_ps(c0,_mm_setr_ps(1.0f,-1.0f,-1.0f,-1.0f));
      _mm_store_ps(&m[0],c0);
      _mm_store_ps(&m[4],c1);
      _mm_store_ps(&m[8],c2);
      _mm_store_ps(&m[12],c3);
    }
    
    //here are operations on lorentz 4 vectors
    LorentzVector operator* (const LorentzVector& o) const{
      float res[4] __attribute__ ((aligned (16)));
      cblas_sgemv(CblasColMajor,
		  CblasNoTrans,
		  4,4,
		  1.0,m,4,
		  o.array(),1,
		  0,res,1); // computes r = m*o
      return LorentzVector(res);
    }
    
    inline void Print() const {
      for( int i = 0; i < 16; ++i ) {
	std::cout << m[i/4*4 + i%4] << ' ';
	if( (i+1)%4 == 0 ) std::cout << std::endl;
      }
    }
  //ClassDef(LorentzRotation,1);
};

}
#endif
