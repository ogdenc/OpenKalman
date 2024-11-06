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
  namespace detail
  {
#ifdef __cpp_lib_span
    template<typename Range, typename Indexible>
    struct index_range_for_impl : std::false_type {};

    template<typename T, std::size_t Extent, typename Indexible>
    struct index_range_for_impl<std::span<T, Extent>, Indexible> 
      : std::bool_constant<(Extent == std::dynamic_extent or Extent >= index_count_v<Indexible>)> {};
#else
    template<typename Range, typename Indexible, typename = void>
    struct index_range_for_impl : std::false_type {};

    template<typename Range, typename Indexible>
    struct index_range_for_impl<Range, Indexible, std::void_t<std::tuple_size<T>::type>>
      : std::bool_constant<std::tuple_size_v<T> >= index_count_v<Indexible>> {};
#endif
  }


  /**
   * \brief Indices is a std::ranges::sized_range of indices that are compatible with \ref indexible object T.
   * \todo Rename to "index_range_for"
   */
  template<typename Indices, typename T>
#if defined(__cpp_lib_ranges) and defined(__cpp_lib_span)
  concept static_range_size =
    indexible<T> and std::ranges::input_range<std::decay_t<Indices>> and 
    index_value<std::ranges::range_value_t<Indices>> and 
    interface::get_component_defined_for<T, T, Indices> and 
    (index_count_v<T> == dynamic_size or
      detail::index_range_for_impl<decltype(std::span{std::declval<std::add_lvalue_reference_t<Indices>>()}), T>::value);
#else
  constexpr bool static_range_size = 
    indexible<T> and index_value<decltype(*std::declval<Indices>().begin())> and 
    interface::get_component_defined_for<T, T, Indices> and 
    (index_count_v<T> == dynamic_size or detail::index_range_for_impl<Indices, T>::value);
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_STATIC_RANGE_SIZE_HPP
