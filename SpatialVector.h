#ifndef __fhep_SpatialVector_h__
#define __fhep_SpatialVector_h__

#include <cmath>
#include <algorithm>

#ifndef __CINT__ // so cint can handle attributes and such, good
#include <smmintrin.h>
#else
typedef float __m128 __attribute__ ((__vector_size__ (16), __may_alias__));
#endif

namespace fhep {

  class SpatialVector {
    
  public:
    // we'll use an __m128 here and leave the lowest packed float as zero all the time
    // 
    union fourvec {
      float v[4];
      __m128 base;
    };
    
  private:
    fourvec _fP4 __attribute__ ((aligned (16))); // [1:3] = cartesian directions
  public:
    //default
    SpatialVector() {
      _fP4.base = _mm_setzero_ps();
    }
    
    // ctor from floats
    SpatialVector(float x, float y, float z) {
      _fP4.base = _mm_setr_ps(0.0,x,y,z);
    }
    
    // ctor from float array
    SpatialVector(const float *carray) {
      memcpy(_fP4.v,carray,4*sizeof(float));
    }
    
    //ctor from intrinsic
    SpatialVector(const __m128& o) {
      _fP4.base = o;
    }
    
    //cctor
    SpatialVector(const SpatialVector& a) {
      memcpy(_fP4.v,a._fP4.v,4*sizeof(float));
    }
    
    inline float x() const { return _fP4.v[1]; }
    inline float y() const { return _fP4.v[2]; }
    inline float z() const { return _fP4.v[3]; }
    
    inline void setX(float& xx) { _fP4.v[1] = xx; }
    inline void setY(float& yy) { _fP4.v[2] = yy; }
    inline void setZ(float& zz) { _fP4.v[3] = zz; }
    
    inline float px()     const { return _fP4.v[1]; }
    inline float py()     const { return _fP4.v[2]; }
    inline float pz()     const { return _fP4.v[3]; }
    
    inline void setPx(float& x) { _fP4.v[1] = x; }
    inline void setPy(float& y) { _fP4.v[2] = y; }
    inline void setPz(float& z) { _fP4.v[3] = z; }
    
    inline float theta() const {
      float pmag = mag();
      return ( pmag == 0.0 ? 0.0 : std::atan2(perp(),_fP4.v[3]) );
    }

    inline float cosTheta() const {
      float pmag = mag();
      return ( pmag == 0.0 ? 1.0 : _fP4.v[3]/pmag );
    }

    inline float phi() const {
      float pmag = mag();
      return ( pmag == 0.0 ? 0.0 : std::atan2(_fP4.v[2],_fP4.v[1]) );
    }
    
    inline float rho() const {
      return mag();
    }

    /*        
    inline void SetTheta(Double_t& theta) {}
    inline void SetPhi(Double_t& phi) {}
    inline void SetRho(Double_t& rho) {}
    */
    
    inline void setXYZ(const float& x, const float& y, const float& z) {
      _fP4.base = _mm_setr_ps(0.0,x,y,z);
    }
    inline void setPxPyPzE(const float& px, const float& py, const float& pz) {
      setXYZT(px,py,pz);
    }    
    
    // this rotates indices to what ROOTy people are used to
    inline float operator[] (const int& i) const {   
      return _fP4.v[(i)%3+1]; 
    }
    
    // 4-vector operations on 4-vectors and scalars
    inline SpatialVector& operator =  (const SpatialVector& o) {
      _fP4.base = o._fP4.base;
      return *this;
    }
    inline SpatialVector& operator = (const float* o) {
      _fP4.base = _mm_load_ps(o);
      return *this;
    }  
    inline SpatialVector  operator +  (const SpatialVector& o) const {
      return SpatialVector(_mm_add_ps(_fP4.base,o._fP4.base));
    }
    inline SpatialVector& operator += (const SpatialVector& o) {
      _fP4.base = _mm_add_ps(_fP4.base,o._fP4.base);
      return *this;
    }
    inline SpatialVector  operator -  (const SpatialVector& o) const {
      return SpatialVector(_mm_sub_ps(_fP4.base,o._fP4.base));
    }
    inline SpatialVector& operator -= (const SpatialVector& o) {
      _fP4.base = _mm_sub_ps(_fP4.base,o._fP4.base);
      return *this;
    }
    inline SpatialVector operator - () const {
      return SpatialVector(_mm_mul_ps(_mm_set1_ps(-1.0),_fP4.base)); // -v[0], -v[1], -v[2], -v[3]
    }
    inline SpatialVector operator* (float a) const {
      return SpatialVector(_mm_mul_ps(_mm_set1_ps(a),_fP4.base)); // a*v[0], a*v[1], a*v[2], a*v[3] 
    }
    inline SpatialVector operator*= (float a) {
      _fP4.base = _mm_mul_ps(_mm_set1_ps(a),_fP4.base); 
      return *this;
    }
    inline float operator* (const SpatialVector& o) const {    
      return _mm_dp_ps(_fP4.base,o._fP4.base,0xe1)[0]; 
      // \sum a_i*b_i, i = 1..3
    }
    
    // return the float* to our 4 vector
    inline const float* array() const { return _fP4.v; }
    
    //invariants and other useful physics things a la ROOT
    inline float mag2() const { 
      return _mm_dp_ps(_fP4.base,_fP4.base,0xe1)[0]; 
      // 0xe1 = 1110 0001, multiply/accumulate all terms in input into lowest float of output
    }
    
    inline float mag() const  {      
      return _mm_sqrt_ss(_mm_dp_ps(_fP4.base,_fP4.base,0xe1))[0];
    }
    
    inline void setmag(const float& s) const {
      float current = mag();
      if(current == 0.0) return;
      current = s/current;
      this->operator*(current); // scale this four vector up to mag s
    }

    inline float perp2() const {
      return _mm_dp_ps(_fP4.base,_fP4.base,0x61)[0];
    }
    inline float perp() const {
      return _mm_sqrt_ss(_mm_dp_ps(_fP4.base,_fP4.base,0x61))[0];
      // x*x + y*y
      // 0x61 = 0110 0001 multiply terms 1,2 in 4vec, accumulate into lowest float in output
    }
    inline float pt() const { return perp(); }
    
    //perp2 that determines component normal to another vector
    inline float perp2(const SpatialVector& p) const {
      float tot = p.mag2();
      float ss  = dot(p);
      float per = mag2();
      per -= (tot > 0.0)*ss*ss/tot; // branching here is more expensive than subtract-multiply
      per *= (per >= 0); // same as if(per < 0) per = 0;
      return per;
    }

    inline float pseudorapidity() const {
      float mag = mag();
      return 0.5*(std::log(mag + _fP4.v[3]) - std::log(mag - _fP4.v[3])); // -(1/2)*log(|p| - z/ |p| + z)
    }
    
    inline float eta() const { return pseudorapidity(); }
        
    inline float dot(const SpatialVector& o) const { return *this*o; }
    
    //logical operations
    inline bool operator==(const SpatialVector& o) const {
      __m128 t = _mm_cmpneq_ps(_fP4.base,o._fP4.base);    
      t = _mm_dp_ps(t,t,0xf1);
      return t[0] == 0.0f;
      // intrinsic returns 0xffffffff in each channel that's not equal
    }
    inline bool operator!=(const SpatialVector& o) const {
      __m128 t = _mm_cmpneq_ps(_fP4.base,o._fP4.base);    
      t = _mm_dp_ps(t,t,0xf1);
      return t[0] != 0.0f;
      // intrinsic returns 0xffffffff in each channel that's not equal
    }
  };

}

#endif
