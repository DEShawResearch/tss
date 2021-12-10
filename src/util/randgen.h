#pragma once
#include "util.hxx"

/* Random123 stuff */
#include <Random123/philox.h>
#include <Random123/u01fixedpt.h>

typedef r123::Philox2x64 r123_type;

/*! \file util/randgen/randgen.hxx

  Provides random number generators which are suitable for used in
  parallel environments, because they are counter-based.

  Defines two classes.  One representing a random value, which
  can be extracted as an integer or a real with various distributions.
  The other is a random value generator, which is counter-based and
  can return the same number if it is not advanced.

 */

  /*! A 'value' returned by the random number generator.  This
    value contains a bunch of random bits and its member functions
    extract those bits so that the result follows the selected
    distribution. */
  struct randval {
    /*! c'tor, we expect only randgen to use this.
      @param v Random123 counter type.
    */
    randval(const r123_type::ctr_type &v) : value(v) {}

    //! Return an unsigned 8 bit int with all 8 bits random.
    uint8_t as_uint8() const { return value[0]; }
    //! Return a non-negative signed 8 bit int (7 bits random).
    int8_t  as_int8() const { return std::abs(int8_t(as_uint8())); }

    //! Return an unsigned 8 bit int with all 16 bits random.
    uint16_t as_uint16() const { return value[0]; }
    //! Return a non-negative signed 8 bit int (15 bits random).
    int16_t  as_int16() const { return std::abs(int16_t(as_uint16())); }

    //! Return an unsigned 8 bit int with all 32 bits random.
    uint32_t as_uint32() const { return value[0]; }
    //! Return a non-negative signed 8 bit int (31 bits random).
    int32_t  as_int32() const { return std::abs(int32_t(as_uint32())); }

    //! Return an unsigned 8 bit int with all 64 bits random.
    uint64_t as_uint64() const { return value[0]; }
   //! Return a non-negative signed 8 bit int (63 bits random).
    int64_t  as_int64() const { return std::abs(int64_t(as_uint64())); }

    //! Return a uniformly distributed float on (0,1)
    float   as_float_oo() const {
      return u01fixedpt_open_open_32_float(as_uint32());
    }
    //! Return a uniformly distributed float on (0,1]
    float   as_float_oc() const {
      return u01fixedpt_open_closed_32_float(as_uint32());
    }
    //! Return a uniformly distributed float on [0,1)
    float   as_float_co() const {
      return u01fixedpt_closed_open_32_float(as_uint32());
    }
    //! Return a uniformly distributed float on [0,1]
    float   as_float_cc() const {
      return u01fixedpt_closed_closed_32_float(as_uint32());
    }

    //! Return a uniformly distributed double on (0,1)
    double   as_double_oo() const {
      return u01fixedpt_open_open_64_double(as_uint64());
    }
    //! Return a uniformly distributed double on (0,1]
    double   as_double_oc() const {
      return u01fixedpt_open_closed_64_double(as_uint64());
    }
    //! Return a uniformly distributed double on [0,1)
    double   as_double_co() const {
      return u01fixedpt_closed_open_64_double(as_uint64());
    }
    //! Return a uniformly distributed double on [0,1]
    double   as_double_cc() const {
      return u01fixedpt_closed_closed_64_double(as_uint64());
    }

    //! we leave the value accessible in case someone
    //! has some dark purpose for it.
    r123_type::ctr_type value;
  };

  /*! Random number generator.  In some ways, this can be
    thought of as a generator of tables of random numbers.
    Different elements of the table can be accessed through
    the 'key' parameter of the \Code{value} member function.
    The \Code{next} function then advanced the generator to
    a completely different random table. */
  struct randgen {

    /*! reset the counter and set the seed.
      @param seed
    */
    void seed(uint64_t seed);

    /*! compute and return the random value corresponding
      to the current randgen state.  The optional key parameter
      allows the used to select alternate values.  For example,
      \Code{value(1)} and \Code{value(2)} are independent random
      values.  If you were generating a random value for each
      particle, for example, you might take key equal to each
      particle's GID.
      @param key additional random value salt.
    */
    randval value(uint64_t key=0) const;

    //! Advance the state counter to the next set of random values.
    randgen &next() { state[0]++; return *this; }

    //! The next().value() combo is very common.
    randval  yield() { return next().value(); }

    //! Reverse the state counter.  Not sure why anyone would use
    //! this, but why not.
    randgen &prev() { state[0]--; return *this; }

    randgen() {}
  private:
    r123_type::ctr_type state; // state[0] = sequence count, state[1] = seed
    friend class desres::oats::access;
    template <class A> void cerealize(A& a) {
      a(state[0])(state[1]);
    }
  };
