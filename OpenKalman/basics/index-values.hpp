/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward definitions for index values.
 */

#ifndef OPENKALMAN_INDEX_VALUES_HPP
#define OPENKALMAN_INDEX_VALUES_HPP

#include <type_traits>

namespace OpenKalman
{
  // ---------------------- //
  //   static_index_value   //
  // ---------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename Z, typename = void>
    struct is_static_index_value : std::false_type {};

    template<typename T, typename Z>
    struct is_static_index_value<T, Z, std::enable_if_t<std::is_convertible<decltype(std::decay_t<T>::value), const Z>::value>>
      : std::bool_constant<std::decay_t<T>::value >= 0 and static_cast<Z>(std::decay_t<T>{}) == static_cast<Z>(std::decay_t<T>::value)> {};
  }
#endif


  /**
   * \brief T is a static index value.
   * \tparam Z the type to which the index must be convertible.
   */
  template<typename T, typename Z = std::size_t>
#ifdef __cpp_concepts
  concept static_index_value = (std::decay_t<T>::value >= 0) and
    std::bool_constant<static_cast<Z>(std::decay_t<T>{}) == static_cast<Z>(std::decay_t<T>::value)>::value;
#else
  constexpr bool static_index_value = detail::is_static_index_value<T, Z>::value;
#endif


  // ----------------------- //
  //   dynamic_index_value   //
  // ----------------------- //

  /**
   * \brief T is a dynamic index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index_value = std::integral<std::decay_t<T>> and std::convertible_to<T, const std::size_t&>;
#else
  template<typename T>
  constexpr bool dynamic_index_value = std::is_integral_v<std::decay_t<T>> and std::is_convertible_v<T, const std::size_t&>;
#endif


  // --------------- //
  //   index_value   //
  // --------------- //

  /**
   * \brief T is an index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept index_value =
#else
  template<typename T>
  constexpr bool index_value =
#endif
    static_index_value<T> or dynamic_index_value<T>;


  // ------------------- //
  //  static_range_size  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_sized_range : std::false_type {};

    template<typename T>
    struct is_sized_range<T, std::void_t<decltype(std::size(std::declval<T>())), decltype(*std::declval<T>().begin())>> : std::true_type {};


    template<typename T, typename = void>
    struct static_range_size_impl : std::false_type {};

    template<typename T>
    struct static_range_size_impl<T, std::void_t<std::integral_constant<std::size_t, std::size(T{})>>> : std::true_type {};
  }
#endif


  /**
   * \brief The static size of a range (e.g., an index range). If this value is not static, the result is OpenKalman::dynamic_size.
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct static_range_size {};


  /**
   * \overload
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<typename T> requires std::ranges::sized_range<std::decay_t<T>> and
    requires { typename std::bool_constant<(std::ranges::size(std::decay_t<T>{}), true)>; }
  struct static_range_size<T> : std::integral_constant<std::size_t, std::ranges::size(std::decay_t<T>{})> {};
#else
  template<typename T>
  struct static_range_size<T, std::enable_if_t<detail::is_sized_range<std::decay_t<T>>::value and
    detail::static_range_size_impl<std::decay_t<T>>::value>>
    : std::integral_constant<std::size_t, std::size(std::decay_t<T>{})> {};
#endif


  /**
   * \overload
   */
#if defined(__cpp_lib_concepts) and defined(__cpp_lib_ranges)
  template<typename T> requires std::ranges::sized_range<std::decay_t<T>> and
    (not requires { typename std::bool_constant<(std::ranges::size(std::decay_t<T>{}), true)>; })
  struct static_range_size<T>
#else
  template<typename T>
  struct static_range_size<T, std::enable_if_t<detail::is_sized_range<std::decay_t<T>>::value and
    (not detail::static_range_size_impl<std::decay_t<T>>::value)>>
#endif
    : std::integral_constant<std::size_t, dynamic_size> {};


  /**
   * \brief helper template for \ref index_range_size.
   */
  template<typename T>
  static constexpr std::size_t static_range_size_v = static_range_size<T>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_VALUES_HPP
