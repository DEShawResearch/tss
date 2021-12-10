#include "randgen.h"   /* For data structures and function prototypes */

randval randgen::value(uint64_t key) const {
  r123_type::key_type k;
  k[0] = key;
  r123_type rng; // Random123 is lame in this way
  return rng(state,k);
}

void randgen::seed(uint64_t seed) {
  state[0] = 0;
  state[1] = seed;
}
