/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2021-2025 Christopher Lee Ogden <ogden@gatech.edu>
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

#ifdef __clang__
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 (__clang_major__ >= 15) // optimal value may be as low as > 10 (ver. 10.0.0)
#elif defined(__GNUC__)
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   (__GNUC__ >= 20) // optimal value may be as low as > 10 (ver. 10.1.0)
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 (__GNUC__ >= 12) // optimal value may be as low as > 10 (ver. 10.1.0)
#else
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS   true
#  define OPENKALMAN_CPP_FEATURE_CONCEPTS_2 true
#endif


#ifdef __cpp_lib_math_constants
#include <numbers>
namespace OpenKalman::numbers { using namespace std::numbers; }
#else
// These re-create the c++20 mathematical constants.
namespace OpenKalman::numbers
{
#ifdef __cpp_lib_concepts
#include <concepts>
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
#include <type_traits>
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


// std::size_t literal similar and equivalent to "uz" literal defined in c++23 standard.
constexpr std::size_t operator ""_uz(unsigned long long x) { return x; };


#ifndef __cpp_lib_integer_comparison_functions
namespace OpenKalman
{
  template<class T, class U>
  constexpr bool cmp_equal(T t, U u) noexcept
  {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
      return t == u;
    else if constexpr (std::is_signed_v<T>)
      return t >= 0 && std::make_unsigned_t<T>(t) == u;
    else
      return u >= 0 && std::make_unsigned_t<U>(u) == t;
  }

  template<class T, class U>
  constexpr bool cmp_not_equal(T t, U u) noexcept
  {
    return !cmp_equal(t, u);
  }

  template<class T, class U>
  constexpr bool cmp_less(T t, U u) noexcept
  {
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>)
      return t < u;
    else if constexpr (std::is_signed_v<T>)
      return t < 0 || std::make_unsigned_t<T>(t) < u;
    else
      return u >= 0 && t < std::make_unsigned_t<U>(u);
  }

  template<class T, class U>
  constexpr bool cmp_greater(T t, U u) noexcept
  {
    return cmp_less(u, t);
  }

  template<class T, class U>
  constexpr bool cmp_less_equal(T t, U u) noexcept
  {
    return !cmp_less(u, t);
  }

  template<class T, class U>
  constexpr bool cmp_greater_equal(T t, U u) noexcept
  {
    return !cmp_less(t, u);
  }
}
#endif


#ifndef __cpp_lib_remove_cvref
namespace OpenKalman
{
  template<typename T>
  struct remove_cvref { using type = std::remove_cv_t<std::remove_reference_t<T>>; };

  template<typename T>
  using remove_cvref_t = typename remove_cvref<T>::type;
}
#endif

namespace OpenKalman::internal
{
  namespace detail
  {
    struct decay_copy_impl final
    {
      template<typename T>
      constexpr std::decay_t<T> operator()(T&& t) const { return std::forward<T>(t); }
    };
  }

  inline constexpr detail::decay_copy_impl decay_copy;
}


#ifndef __cpp_lib_bounded_array_traits
namespace OpenKalman
{
  template<typename T>
  struct is_bounded_array : std::false_type {};

  template<typename T, std::size_t N>
  struct is_bounded_array<T[N]> : std::true_type {};

  template<typename T>
  constexpr bool is_bounded_array_v = is_bounded_array<T>::value;
}
#endif


namespace OpenKalman::internal
{
  namespace detail
  {
    template<std::size_t i, typename T, typename = void>
    struct member_get_is_defined : std::false_type {};

    template<std::size_t i, typename T>
    struct member_get_is_defined<i, T, std::void_t<decltype(std::declval<T>().template get<i>())>> : std::true_type {};


    namespace func_get_def
    {
      using std::get;

      template<std::size_t i, typename T, typename = void>
      struct function_get_is_defined : std::false_type {};

      template<std::size_t i, typename T>
      struct function_get_is_defined<i, T, std::void_t<decltype(get<i>(std::declval<T>()))>> : std::true_type {};
    }

    using func_get_def::function_get_is_defined;


    template<std::size_t i>
    struct get_impl
    {
      template<typename T, std::enable_if_t<member_get_is_defined<i, T>::value or function_get_is_defined<i, T>::value, int> = 0>
      constexpr decltype(auto)
      operator() [[nodiscard]] (T&& t) const
      {
        using std::get;
        if constexpr (member_get_is_defined<i, T>::value)
          return std::forward<T>(t).template get<i>();
        else
          return get<i>(std::forward<T>(t));
      }
    };
  }

  /**
   * \internal
   * \brief This is a placeholder for a more general <code>std::get</code> function that might be added to the standard library, possibly by another name.
   */
  template<std::size_t i>
  inline constexpr detail::get_impl<i> generalized_std_get;

}


#endif //OPENKALMAN_LANGUAGE_FEATURES_HPP
