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
  // --------------------------------------------- //
  //   static_index_value, static_index_value_of   //
  // --------------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_static_index_value : std::false_type {};

    template<typename T>
    struct is_static_index_value<T, std::enable_if_t<std::is_convertible<
      decltype(std::decay_t<T>::value), const std::size_t>::value and std::is_default_constructible_v<std::decay_t<T>>>>
      : std::bool_constant<std::is_convertible_v<T, const decltype(std::decay_t<T>::value)> and std::decay_t<T>::value >= 0> {};
  }
#endif


  /**
   * \brief T is a static index value.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept static_index_value = std::convertible_to<decltype(std::decay_t<T>::value), const std::size_t> and
    std::convertible_to<T, const decltype(std::decay_t<T>::value)> and
    (std::decay_t<T>::value >= 0) and std::default_initializable<std::decay_t<T>>;
#else
  constexpr bool static_index_value = detail::is_static_index_value<T>::value;
#endif


  /**
   * \brief The numerical value of a \ref static_index_value.
   * \details If T is not a static, compile-time constant, the result is \ref dynamic_size.
   * \todo Replace this in all its instances by direct conversion to the integral type?
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct static_index_value_of : std::integral_constant<std::size_t, dynamic_size> {};


#ifdef __cpp_concepts
  template<static_index_value T>
  struct static_index_value_of<T>
#else
  template<typename T>
  struct static_index_value_of<T, std::enable_if_t<static_index_value<T>>>
#endif
    : std::integral_constant<std::size_t, std::decay_t<T>::value> {};


  /**
   * \brief Helper template for \ref static_index_value_of.
   */
  template<typename T>
  constexpr auto static_index_value_of_v = static_index_value_of<T>::value;


  // ----------------------- //
  //   dynamic_index_value   //
  // ----------------------- //

  /**
   * \brief T is a dynamic index value.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept dynamic_index_value = std::integral<std::decay_t<T>>;
#else
  template<typename T>
  constexpr bool dynamic_index_value = std::is_integral_v<std::decay_t<T>>;
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


} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_VALUES_HPP
