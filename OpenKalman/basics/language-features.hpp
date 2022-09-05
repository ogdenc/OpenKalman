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


#ifdef __cpp_lib_math_constants
#include <numbers>
namespace OpenKalman::numbers { using std::numbers; }
#else
// These re-create the c++20 mathematical constants.
namespace OpenKalman::numbers
{
#ifdef __cpp_concepts
  template<std::floating_point T> inline constexpr T e_v = 2.718281828459045235360287471352662498L;
  template<std::floating_point T> inline constexpr T log2e_v = 1.442695040888963407359924681001892137L;
  template<std::floating_point T> inline constexpr T log10e_v = 0.434294481903251827651128918916605082L;
  template<std::floating_point T> inline constexpr T pi_v = 3.141592653589793238462643383279502884L;
  template<std::floating_point T> inline constexpr T inv_pi_v = 0.318309886183790671537767526745028724L;
  template<std::floating_point T> inline constexpr T inv_sqrtpi_v = 0.564189583547756286948079451560772586L;
  template<std::floating_point T> inline constexpr T ln2_v = 0.693147180559945309417232121458176568L;
  template<std::floating_point T> inline constexpr T ln10_v = 2.302585092994045684017991454684364208L;
  template<std::floating_point T> inline constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;
  template<std::floating_point T> inline constexpr T sqrt3_v = 1.732050807568877293527446341505872367L;
  template<std::floating_point T> inline constexpr T inv_sqrt3_v = 0.577350269189625764509148780501957456L;
  template<std::floating_point T> inline constexpr T egamma_v = 0.577215664901532860606512090082402431L;
  template<std::floating_point T> inline constexpr T phi_v = 1.618033988749894848204586834365638118L;
#else
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T e_v = 2.718281828459045235360287471352662498L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T log2e_v = 1.442695040888963407359924681001892137L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T log10e_v = 0.434294481903251827651128918916605082L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T pi_v = 3.141592653589793238462643383279502884L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_pi_v = 0.318309886183790671537767526745028724L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_sqrtpi_v = 0.564189583547756286948079451560772586L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T ln2_v = 0.693147180559945309417232121458176568L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T ln10_v = 2.302585092994045684017991454684364208L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T sqrt2_v = 1.414213562373095048801688724209698079L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T sqrt3_v = 1.732050807568877293527446341505872367L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T inv_sqrt3_v = 0.577350269189625764509148780501957456L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T egamma_v = 0.577215664901532860606512090082402431L;
  template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0> inline constexpr T phi_v = 1.618033988749894848204586834365638118L;
#endif

  inline constexpr double e = e_v<double>;
  inline constexpr double log2e = log2e_v<double>;
  inline constexpr double log10e = log10e_v<double>;
  inline constexpr double pi = pi_v<double>;
  inline constexpr double inv_pi = inv_pi_v<double>;
  inline constexpr double inv_sqrtpi = inv_sqrtpi_v<double>;
  inline constexpr double ln2 = ln2_v<double>;
  inline constexpr double ln10 = ln10_v<double>;
  inline constexpr double sqrt2 = sqrt2_v<double>;
  inline constexpr double sqrt3 = sqrt3_v<double>;
  inline constexpr double inv_sqrt3 = inv_sqrt3_v<double>;
  inline constexpr double egamma = egamma_v<double>;
  inline constexpr double phi = phi_v<double>;
}
#endif


#endif //OPENKALMAN_LANGUAGE_FEATURES_HPP
