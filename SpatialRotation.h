#ifndef __fhep_SpatialRotation_h__
#define __fhep_SpatialRotation_h__

#include "SpatialVector.h"
#include "LorentzVector.h"

#include <vecLib/cblas.h>
#include <iostream>
#include <cmath>

#ifndef __CINT__
#include <smmintrin.h>
#endif

namespace fhep {
  // spatial rotation is simply the SO(3) sub matrix of a lorentz rotation
  // treat it as such.
  class SpatialRotation {
    
  private:
    float m[16] __attribute__ ((aligned (16))); // column major
    
    /*
      tt tx ty tz
      xt xx xy xz
      yt yx yy yz
      zt zx zy zz
    */
    
  public:
    
    SpatialRotation() { // initialize to identity
      memset(m,0,16*sizeof(float));
      m[0] = m[4] = m[8] = m[12] = 1.0f;
    }
    
    SpatialRotation(const SpatialRotation& o) {
      memcpy(m,o.m,16*sizeof(float));
    }
    
    SpatialRotation(const float* o) {
      memcpy(m,o,16*sizeof(float));
    }
    
    SpatialRotation( float xx, float xy, float xz, 
		     float yx, float yy, float yz, 
		     float zx, float zy, float zz ) {
      float tmp[16] = {1.0f,0.0f,0.0f,0.0f,0.0f,xx,yx,zx,0.0f,xy,yy,zy,0.0f,xz,yz,zz};
      memcpy(m,tmp,16*sizeof(float));
    }
    
    void setToIdentity() {
      memset(m,0,16*sizeof(float));
      m[4] = m[8] = m[12] = 1.0f;
    }

    inline SpatialRotation& operator= (const SpatialRotation& o) {
      memcpy(m,o.m,16*sizeof(float));
      return *this;
    }
    // I am not going to implement the "Transform" functions
    // they are redundant
    // here are operations on other matrices
    inline SpatialRotation operator* (const SpatialRotation& o) const {
      float res[16] __attribute__ ((aligned (16)));
      cblas_sgemm(CblasColMajor,
		  CblasNoTrans,
		  CblasNoTrans,
		  4,4,4,
		  1.0,m,4,
		  o.m,4,
		  0,res,4); // computes c = m*o for a 4x4 matrix of floats, putting the result into m
      return SpatialRotation(res);
    }
    inline SpatialRotation& operator*= (const SpatialRotation& o) {
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
    SpatialRotation operator~() const {
      float res[16] __attribute__ ((aligned (16)));
      __m128 c0 = _mm_load_ps(&m[0]);
      __m128 c1 = _mm_load_ps(&m[4]);
      __m128 c2 = _mm_load_ps(&m[8]);
      __m128 c3 = _mm_load_ps(&m[12]);
      
      _MM_TRANSPOSE4_PS(c0,c1,c2,c3);
      
      _mm_store_ps(&res[0],c0);
      _mm_store_ps(&res[4],c1);
      _mm_store_ps(&res[8],c2);
      _mm_store_ps(&res[12],c3);
      return SpatialRotation(res);
    }
    inline SpatialRotation inverse() const { return operator~(); }
    
    void invert() {
      __m128 c0 = _mm_load_ps(&m[0]);
      __m128 c1 = _mm_load_ps(&m[4]);
      __m128 c2 = _mm_load_ps(&m[8]);
      __m128 c3 = _mm_load_ps(&m[12]);
      
      _MM_TRANSPOSE4_PS(c0,c1,c2,c3);
      
      _mm_store_ps(&m[0],c0);
      _mm_store_ps(&m[4],c1);
      _mm_store_ps(&m[8],c2);
      _mm_store_ps(&m[12],c3);
    }
    
    //here are operations on 3 vectors
    SpatialVector operator* (const SpatialVector& o) const{
      float res[4] __attribute__ ((aligned (16)));
      cblas_sgemv(CblasColMajor,
		  CblasNoTrans,
		  4,4,
		  1.0,m,4,
		  o.array(),1,
		  0,res,1); // computes r = m*o
      return SpatialVector(res);
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
    
    const float* array() const { return m; }

    inline void Print() const {
      for( int i = 0; i < 16; ++i ) {
	std::cout << m[i/4*4 + i%4] << ' ';
	if( (i+1)%4 == 0 ) std::cout << std::endl;
      }
    }
  //ClassDef(SpatialRotation,1);
};

}
#endif
