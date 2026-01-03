/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024-2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref index_collection_for.
 */

#ifndef OPENKALMAN_INDEX_COLLECTION_FOR_HPP
#define OPENKALMAN_INDEX_COLLECTION_FOR_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename Indices, typename T, typename Seq = std::make_index_sequence<collections::size_of_v<Indices>>>
    struct index_collection_for_iter : std::false_type {};

    template<typename Indices, typename T, std::size_t...ix>
    struct index_collection_for_iter<Indices, T, std::index_sequence<ix...>>
      : std::bool_constant<(... and (not values::size_compares_with<
            collections::collection_element_t<ix, Indices>,
            index_dimension_of<T, ix>,
            &stdex::is_gteq
          >))> {};


#ifndef __cpp_concepts
    template<typename Indices, typename Indexible, typename = void>
    struct index_collection_for_impl : std::true_type {};

    template<typename Indices, typename Indexible>
    struct index_collection_for_impl<Indices, Indexible, std::enable_if_t<
      values::fixed_value_compares_with<collections::size_of<Indices>, stdex::dynamic_extent, &stdex::is_neq>>>
        : index_collection_for_iter<Indices, Indexible> {};
#endif
  }


  /**
   * \brief Indices is a \ref collections::index "collection of indices" that are compatible with \ref indexible object T.
   * \details This performs static bounds checking.
   */
  template<typename Indices, typename T>
#ifdef __cpp_concepts
  concept index_collection_for =
    collections::index<Indices> and
    indexible<T> and
    (not values::size_compares_with<collections::size_of<Indices>, index_count<T>, &stdex::is_lt>) and
    (values::fixed_value_compares_with<collections::size_of<Indices>, stdex::dynamic_extent> or
      detail::index_collection_for_iter<Indices, T>::value);
#else
  constexpr bool index_collection_for =
    collections::index<Indices> and
    indexible<T> and
    (not values::size_compares_with<collections::size_of<Indices>, index_count<T>, &stdex::is_lt>) and
    detail::index_collection_for_impl<Indices, T>::value;
#endif

}

#endif
