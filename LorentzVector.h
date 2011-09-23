#ifndef __fhep_LorentzVector_h__
#define __fhep_LorentzVector_h__

#include <cmath>
#include <algorithm>

#ifndef __CINT__ // so cint can handle attributes and such, good
#include <smmintrin.h>
#else
typedef float __m128 __attribute__ ((__vector_size__ (16), __may_alias__));
#endif

namespace fhep {

  class LorentzVector {
    
  public:
    union fourvec {
      float v[4];
      __m128 base;
    };
    
  private:
    fourvec _fP4 __attribute__ ((aligned (16))); // [1:3] = spacelike part, [0] = timelike
    static const __m128 metric __attribute__ ((aligned (16)));  
    
  public:
    //default
    LorentzVector() {
      _fP4.base = _mm_setzero_ps();
    }
    
    // ctor from floats
    LorentzVector(float x, float y, 
		  float z, float t) {
      _fP4.base = _mm_setr_ps(t,x,y,z);
    }
    
    // ctor from float array
    LorentzVector(const float *carray) {
      memcpy(_fP4.v,carray,4*sizeof(float));
    }
    
    //ctor from intrinsic
    LorentzVector(const __m128& o) {
      _fP4.base = o;
    }
    
    //cctor
    LorentzVector(const LorentzVector& a) {
      memcpy(_fP4.v,a._fP4.v,4*sizeof(float));
    }
    
    inline float x() const { return _fP4.v[1]; }
    inline float y() const { return _fP4.v[2]; }
    inline float z() const { return _fP4.v[3]; }
    inline float t() const { return _fP4.v[0]; }
    
    inline void setX(float& xx) { _fP4.v[1] = xx; }
    inline void setY(float& yy) { _fP4.v[2] = yy; }
    inline void setZ(float& zz) { _fP4.v[3] = zz; }
    inline void setT(float& tt) { _fP4.v[0] = tt; }
    
    inline float px()     const { return _fP4.v[1]; }
    inline float py()     const { return _fP4.v[2]; }
    inline float pz()     const { return _fP4.v[3]; }
    inline float e()      const { return _fP4.v[0]; }
    inline float energy() const { return _fP4.v[0]; }
    
    inline void setPx(float& x) { _fP4.v[1] = x; }
    inline void setPy(float& y) { _fP4.v[2] = y; }
    inline void setPz(float& z) { _fP4.v[3] = z; }
    inline void setE(float& e)  { _fP4.v[0] = e; }
    
    /*
      inline IVector3 Vect() const { 
      return IVector3(_fP4.v[1],_fP4.v[2],_fP4.v[3]); 
      }
      // IVector3 should just be a __m128 with time component = 0
      // this way we can just use a scalar instruction to set the spatial component
      inline void SetVect(const IVector3& vect3) {
      _fP4.v[1] = vect3.X(); _fP4.v[2] = vect3.Y(); _fP4.v[3] = vect3.Z();
      }
      
      inline Double_t Theta() const {}
      inline Double_t CosTheta() const {}
      inline Double_t Phi() const {}
      inline Double_t Rho() const {}
      
      inline void SetTheta(Double_t& theta) {}
      inline void SetPhi(Double_t& phi) {}
      inline void SetRho(Double_t& rho) {}
    */
    
    inline void setXYZT(const float& x, const float& y,
			const float& z, const float& t) {
      _fP4.base = _mm_setr_ps(t,x,y,z);
    }
    inline void setPxPyPzE(const float& px, const float& py,
			   const float& pz, const float& e) {
      setXYZT(px,py,pz,e);
    }
    
    inline void setXYZM(const float& px, const float& py,
			const float& pz, const float& m) {  
      float zero(0.0);
      __m128 e2 = _mm_setr_ps(m,px,py,pz); // starts off as vector of m and \vec{p}, becomes energy^2
      __m128 flip = _mm_setr_ps(-1.0f,1.0f,1.0f,1.0f);
      e2 = ( m >= 0.0f ? _mm_dp_ps(e2,e2,0xf1) : _mm_dp_ps(_mm_mul_ps(flip,e2),e2,0xf1) );
      float etemp = std::max(e2[0],zero);
      etemp = std::sqrt(etemp);
      setXYZT( px, py, pz, etemp );    
    }
    
    // this rotates indices to what ROOTy people are used to
    inline float operator[] (const int& i) const {   
      return _fP4.v[(i+1)%4]; 
    }
    
    // 4-vector operations on 4-vectors and scalars
    inline LorentzVector& operator =  (const LorentzVector& o) {
      _fP4.base = o._fP4.base;
      return *this;
    }
    inline LorentzVector& operator = (const float* o) {
      _fP4.base = _mm_load_ps(o);
      return *this;
    }  
    inline LorentzVector  operator +  (const LorentzVector& o) const {
      return LorentzVector(_mm_add_ps(_fP4.base,o._fP4.base));
    }
    inline LorentzVector& operator += (const LorentzVector& o) {
      _fP4.base = _mm_add_ps(_fP4.base,o._fP4.base);
      return *this;
    }
    inline LorentzVector  operator -  (const LorentzVector& o) const {
      return LorentzVector(_mm_sub_ps(_fP4.base,o._fP4.base));
    }
    inline LorentzVector& operator -= (const LorentzVector& o) {
      _fP4.base = _mm_sub_ps(_fP4.base,o._fP4.base);
      return *this;
    }
    inline LorentzVector operator - () const {
      return LorentzVector(_mm_mul_ps(_mm_set1_ps(-1.0),_fP4.base)); // -v[0], -v[1], -v[2], -v[3]
    }
    inline LorentzVector operator* (float a) const {
      return LorentzVector(_mm_mul_ps(_mm_set1_ps(a),_fP4.base)); // a*v[0], a*v[1], a*v[2], a*v[3] 
    }
    inline LorentzVector operator*= (float a) {
      _fP4.base = _mm_mul_ps(_mm_set1_ps(a),_fP4.base); 
      return *this;
    }
    inline float operator* (const LorentzVector& o) const {    
      return _mm_dp_ps(_mm_mul_ps(metric,_fP4.base),o._fP4.base,0xf1)[0]; 
      // a^mu g_mu_nu b_nu
    }
    
    // return the float* to our 4 vector
    inline const float* array() const { return _fP4.v; }
    
    //invariants and other useful physics things a la ROOT
    inline float mag2() const { 
      return _mm_dp_ps(_mm_mul_ps(metric,_fP4.base),_fP4.base,0xf1)[0]; 
      // 0xf1 = 1111 0001, multiply/accumulate all terms in input into lowest float of output
    }
    inline float m2() const { return mag2(); }
    
    inline float mag() const  {
      __m128 dp = _mm_dp_ps(_mm_mul_ps(metric,_fP4.base),_fP4.base,0xf1);
      if(_mm_cmpge_ss(dp,_mm_setzero_ps())[0]) {
	return _mm_sqrt_ss(dp)[0];
      }
      return -_mm_sqrt_ss(_mm_mul_ss(dp,_mm_set1_ps(-1.0f)))[0];
    }
    inline float m() const {
      return mag();
    }
    inline float p() const {
      return _mm_sqrt_ss(_mm_dp_ps(_fP4.base,_fP4.base,0xe1))[0];
    }
    inline float mt2() const {
      return _mm_dp_ps(_mm_mul_ps(_fP4.base,metric),_fP4.base,0x91)[0]; 
      // e*e - z*z
      // 0x91 = 1001 0001 multiply/accumulate all terms in input into lowest float of output
    }
    inline float mt() const {
      return ( _fP4.v[0]*_fP4.v[0] < _fP4.v[3]*_fP4.v[3] ? -std::sqrt( -mt2() ) : std::sqrt( mt2()) );
    }
    inline float beta() const {
      __m128 ie4 = _mm_set1_ps(1.0/_fP4.v[0]);
      __m128 v = _mm_mul_ps(_fP4.base,ie4);
      return _mm_sqrt_ss(_mm_dp_ps(v,v,0xe1))[0];
      // |p|/E
      // 0xe1 = 1110 0001 multiply/accumulate spacelike terms and push to lowest float of result
    }
    inline float gamma() const { // think about how to make this better
      __m128 b = _mm_mul_ps(_fP4.base,_mm_set1_ps(1.0/_fP4.v[0]));
    return 1.0f/std::sqrt(1.0f-_mm_dp_ps(b,b,0xe1)[0]);
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
    
    inline float pseudorapidity() const {
      float mag = _mm_sqrt_ss(_mm_dp_ps(_fP4.base,_fP4.base,0xe1))[0];
      return 0.5*(std::log(mag + _fP4.v[3]) - std::log(mag - _fP4.v[3])); // -(1/2)*log(|p| - z/ |p| + z)
    }
    
    inline float eta() const { return pseudorapidity(); }
    
    inline float rapidity() const {
      return 0.5*(std::log(_fP4.v[0] + _fP4.v[3]) - std::log(_fP4.v[0] - _fP4.v[3])); // -(1/2)*log(E - z/ E + z)
    }
    
    inline float rap() const {
      return rapidity();
    }
    
    inline float dot(const LorentzVector& o) const { return *this*o; }
    
    //logical operations
    inline bool operator==(const LorentzVector& o) const {
      __m128 t = _mm_cmpneq_ps(_fP4.base,o._fP4.base);    
      t = _mm_dp_ps(t,t,0xf1);
      return t[0] == 0.0f;
      // intrinsic returns 0xffffffff in each channel that's not equal
    }
    inline bool operator!=(const LorentzVector& o) const {
      __m128 t = _mm_cmpneq_ps(_fP4.base,o._fP4.base);    
      t = _mm_dp_ps(t,t,0xf1);
      return t[0] != 0.0f;
      // intrinsic returns 0xffffffff in each channel that's not equal
    }
  };
  
#ifndef __CINT__
const __m128 LorentzVector::metric = __extension__ (__m128){1.0f, -1.0f, -1.0f, -1.0f};
#else
const __m128 LorentzVector::metric;
#endif

}

#endif
