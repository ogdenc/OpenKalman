/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref static_range_size.
 */

#ifndef OPENKALMAN_STATIC_RANGE_SIZE_HPP
#define OPENKALMAN_STATIC_RANGE_SIZE_HPP

#include <type_traits>

namespace OpenKalman
{
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

#endif //OPENKALMAN_STATIC_RANGE_SIZE_HPP
