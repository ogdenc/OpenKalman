/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definitions implementing features of the c++ ranges library for compatibility.
 *
 * \dir ranges
 * \internal
 * \brief std::ranges definitions for compatibility with c++17 or other legacy versions of c++.
 */

#ifndef OPENKALMAN_RANGES_RANGE_CONCEPTS_HPP
#define OPENKALMAN_RANGES_RANGE_CONCEPTS_HPP

#include "basics/compatibility/language-features.hpp"
#include "basics/compatibility/iterator.hpp"
#include "range-access.hpp"

namespace OpenKalman::stdcompat::ranges
{
#ifdef __cpp_lib_ranges
  using std::ranges::range;
  using std::ranges::borrowed_range;
  using std::ranges::sized_range;
  using std::ranges::input_range;
  using std::ranges::output_range;
  using std::ranges::forward_range;
  using std::ranges::bidirectional_range;
  using std::ranges::random_access_range;
  using std::ranges::common_range;
  using std::ranges::range_size_t;
  using std::ranges::range_difference_t;
  using std::ranges::range_value_t;
  using std::ranges::range_reference_t;
  using std::ranges::range_rvalue_reference_t;
  //using std::ranges::range_common_reference_t; // Not available in certain c++23 versions of CGG and clang
#if __cplusplus >= 202302L
  using std::ranges::range_const_reference_t;
#endif

#else
  // ---
  // range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_range : std::false_type {};

    template<typename T>
    struct is_range<T, std::void_t<iterator_t<T>, sentinel_t<T>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool range = detail::is_range<T>::value;


  // ---
  // borrowed_range
  // ---

  namespace detail_borrowed_range
  {
    template<typename R, typename = void>
    struct is_borrowed_range : std::false_type {};

    template<typename R>
    struct is_borrowed_range<R, std::enable_if_t<range<R> and
      (std::is_lvalue_reference_v<R> or enable_borrowed_range<stdcompat::remove_cvref_t<R>>)>> : std::true_type {};
  }

  template<typename T>
  inline constexpr bool borrowed_range = detail_borrowed_range::is_borrowed_range<T>::value;


  // ---
  // sized_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_sized_range : std::false_type {};

    template<typename T>
    struct is_sized_range<T, std::void_t<decltype(stdcompat::ranges::size(std::declval<T&>()))>> : std::true_type {};

  }

  template<typename T>
  inline constexpr bool sized_range = range<T> and detail::is_sized_range<T>::value;


  // ---
  // input_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct has_input_iterator : std::false_type {};

    template<typename T>
    struct has_input_iterator<T, std::enable_if_t<input_iterator<iterator_t<T>>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool input_range = range<T> and detail::has_input_iterator<T>::value;


  // ---
  // output_range
  // ---

  namespace detail
  {
    template<typename R, typename T, typename = void>
    struct has_output_iterator : std::false_type {};

    template<typename R, typename T>
    struct has_output_iterator<R, T, std::enable_if_t<output_iterator<iterator_t<R>, T>>> : std::true_type {};
  }


  template<typename R, typename T>
  inline constexpr bool output_range = range<R> and detail::has_output_iterator<R, T>::value;


  // ---
  // forward_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct has_forward_iterator : std::false_type {};

    template<typename T>
    struct has_forward_iterator<T, std::enable_if_t<forward_iterator<iterator_t<T>>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool forward_range = input_range<T> and detail::has_forward_iterator<T>::value;


  // ---
  // bidirectional_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct has_bidirectional_iterator : std::false_type {};

    template<typename T>
    struct has_bidirectional_iterator<T, std::enable_if_t<bidirectional_iterator<iterator_t<T>>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool bidirectional_range = forward_range<T> and detail::has_bidirectional_iterator<T>::value;


  // ---
  // random_access_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct has_random_access_iterator : std::false_type {};

    template<typename T>
    struct has_random_access_iterator<T, std::enable_if_t<random_access_iterator<iterator_t<T>>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool random_access_range = bidirectional_range<T> and detail::has_random_access_iterator<T>::value;


  // ---
  // common_range
  // ---

  namespace detail
  {
    template<typename T, typename = void>
    struct is_common_range : std::false_type {};

    template<typename T>
    struct is_common_range<T, std::enable_if_t<std::is_same_v<iterator_t<T>, sentinel_t<T>>>> : std::true_type {};
  }


  template<typename T>
  inline constexpr bool common_range = range<T> and detail::is_common_range<T>::value;



  // ---
  // Range primitives
  // ---

  template<typename R, std::enable_if_t<sized_range<R>, int> = 0>
  using range_size_t = decltype(size(std::declval<R&>()));

  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_difference_t = iter_difference_t<iterator_t<R>>;

  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_value_t = iter_value_t<iterator_t<R>>;

  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_reference_t = iter_reference_t<iterator_t<R>>;

  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_rvalue_reference_t =  iter_rvalue_reference_t<iterator_t<R>>;
#endif


  // range_common_reference_t is not available in at least some c++23 versions of GCC or clang
  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_common_reference_t =  iter_common_reference_t<iterator_t<R>>;


#if __cplusplus < 202302L
  template<typename R, std::enable_if_t<range<R>, int> = 0>
  using range_const_reference_t =  iter_const_reference_t<iterator_t<R>>;
#endif

}

#endif
