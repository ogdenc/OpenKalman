/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
    template<typename Indices, typename = void>
    struct is_sized_range : std::false_type {};

    template<typename Indices>
    struct is_sized_range<Indices, std::void_t<
      decltype(std::size(std::declval<Indices>())), decltype(*std::declval<Indices>().begin())>> : std::true_type {};


    template<typename Indices, std::size_t N, typename = void>
    struct static_range_size_impl : std::false_type {};

    template<typename Indices, std::size_t N>
    struct static_range_size_impl<Indices, std::void_t<
      decltype(std::bool_constant<std::size(std::declval<Indices>()) >= N>)>> : std::true_type {};
  }
#endif


  /**
   * \brief Indices is a std::ranges::sized_range of indices that are compatible with \ref indexible object T.
   * \todo Rename to "index_range_for"
   */
  template<typename Indices, typename T>
#ifdef __cpp_lib_ranges
  concept static_range_size =
    indexible<T> and std::ranges::input_range<std::decay_t<Indices>> and 
    index_value<std::ranges::range_value_t<Indices>> and 
    interface::get_component_defined_for<T, T, Indices> and 
    (index_count_t<T> == dynamic_size or not std::ranges::sized_range<std::decay_t<Indices>> or 
      requires(std::decay_t<Indices> indices) { requires std::ranges::size(indices) >= index_count_t<T>; }) 
  constexpr bool static_range_size = 
    indexible<T> and index_value<decltype(*std::declval<Indices>().begin())> and 
    index_value<std::ranges::range_value_t<Indices>> and 
    interface::get_component_defined_for<T, T, Indices> and 
    (index_count_t<T> == dynamic_size or not detail::is_sized_range<std::decay_t<Indices>>::value or 
      detail::static_range_size_impl<std::decay_t<Indices>, index_count_t<T>>::value>)
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_STATIC_RANGE_SIZE_HPP
