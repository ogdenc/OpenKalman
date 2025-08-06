/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref index_range_for.
 */

#ifndef OPENKALMAN_INDEX_RANGE_FOR_HPP
#define OPENKALMAN_INDEX_RANGE_FOR_HPP

#include <type_traits>
#ifdef __cpp_lib_ranges
#include <ranges>
#else
#include "basics/compatibility/ranges.hpp"
#endif

namespace OpenKalman
{
#ifndef __cpp_lib_ranges
  namespace detail
  {
    template<typename Indices, typename Indexible, typename = void>
    struct index_range_for_impl_it : std::false_type {};

    template<typename Indices, typename Indexible>
    struct index_range_for_impl_it<Indices, Indexible, std::enable_if_t<values::index<stdcompat::ranges::iterator_t<Indices>>>>
        : std::true_type {};


    template<typename Indices, typename Indexible, typename = void>
    struct index_range_for_impl : std::false_type {};

    template<typename Indices, typename Indexible>
    struct index_range_for_impl<Indices, Indexible, std::enable_if_t<
      collections::size_of<Indices>::value == dynamic_size or index_count<Indexible>::value == dynamic_size or
      collections::size_of<Indices>::value >= index_count<Indexible>::value>>
        : std::true_type {};

  }
#endif


  /**
   * \brief Indices is a std::ranges::sized_range of indices that are compatible with \ref indexible object T.
   */
  template<typename Indices, typename T>
#ifdef __cpp_lib_ranges
  concept index_range_for =
    indexible<T> and std::ranges::input_range<std::decay_t<Indices>> and 
    values::index<std::ranges::range_value_t<Indices>> and
    interface::get_component_defined_for<T, T, Indices> and
    (collections::size_of_v<Indices> == dynamic_size or index_count_v<T> == dynamic_size or
      collections::size_of_v<Indices> >= index_count_v<T>);
#else
  constexpr bool index_range_for =
    indexible<T> and
    detail::index_range_for_impl_it<Indices, T>::value and
    interface::get_component_defined_for<T, T, Indices> and 
    detail::index_range_for_impl<Indices, T>::value;
#endif

} // namespace OpenKalman

#endif //OPENKALMAN_INDEX_RANGE_FOR_HPP
