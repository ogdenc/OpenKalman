/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions relating to the availability of c++ language features.
 */

#ifndef OPENKALMAN_LANGUAGE_FEATURES_HPP
#define OPENKALMAN_LANGUAGE_FEATURES_HPP

#include <type_traits>

#ifdef __cpp_concepts
#include <concepts>
#endif


#ifdef __clang__
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS             true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2           false // clang versions known to cause bugs: 10.0.0
#elif defined(__GNUC__)
#  if __GNUC__ > 100
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS           true
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS_2         true
#  else
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS           false // GCC versions known to cause bugs: 10.1.0
#    define OPENKALMAN_CPP_FEATURE_CONCEPTS_2         false // GCC versions known to cause bugs: 10.1.0
#  endif
#else
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS             true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2           true
#endif


#if __cplusplus > 201907L
#include <numbers>
#endif

#ifndef __cpp_lib_math_constants
// These are re-creations of some of the c++20 standard constants, if they are not already defined.
namespace std::numbers
{
  template<typename T>
  constexpr T pi_v = 3.141592653589793238462643383279502884L;

  constexpr double pi = pi_v<double>;

  template<typename T>
  constexpr T log2e_v = 1.442695040888963407359924681001892137L;

  constexpr double log2e = log2e_v<double>;

  template<typename T>
  constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;

  constexpr double sqrt2 = sqrt2_v<double>;
}
#endif


#endif //OPENKALMAN_LANGUAGE_FEATURES_HPP
