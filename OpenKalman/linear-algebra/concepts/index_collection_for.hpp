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

#include "coordinates/coordinates.hpp"
#include "linear-algebra/interfaces/library-interfaces-defined.hpp"
#include "linear-algebra/traits/index_count.hpp"
#include "linear-algebra/traits/index_dimension_of.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<typename Indices, typename Indexible, typename Seq = std::make_index_sequence<collections::size_of_v<Indices>>>
    struct index_collection_for_iter : std::false_type {};

    template<typename Indices, typename Indexible, std::size_t...ix>
    struct index_collection_for_iter<Indices, Indexible, std::index_sequence<ix...>>
      : std::bool_constant<(... and (values::size_compares_with<
            collections::collection_element_t<ix, Indices>,
            index_dimension_of<Indexible, ix>,
            &stdcompat::is_lt,
            applicability::permitted
          >))> {};


#ifndef __cpp_concepts
    template<typename Indices, typename Indexible, typename = void>
    struct index_collection_for_impl : std::true_type {};

    template<typename Indices, typename Indexible>
    struct index_collection_for_impl<Indices, Indexible, std::enable_if_t<
      values::fixed_value_compares_with<collections::size_of<Indices>, dynamic_size, &stdcompat::is_neq>>>
        : index_collection_for_iter<Indices, Indexible> {};
#endif
  }


  /**
   * \brief Indices is a std::ranges::sized_range of indices that are compatible with \ref indexible object T.
   */
  template<typename Indices, typename T>
#ifdef __cpp_concepts
  concept index_collection_for =
    indexible<T> and
    collections::index<Indices> and
    interface::get_component_defined_for<T, T, Indices> and
    values::size_compares_with<collections::size_of<Indices>, index_count<T>, &stdcompat::is_gteq, applicability::permitted> and
    (values::fixed_value_compares_with<collections::size_of<Indices>, dynamic_size, &std::is_eq> or
      detail::index_collection_for_iter<Indices, T>::value);
#else
  constexpr bool index_collection_for =
    indexible<T> and
    collections::index<Indices> and
    interface::get_component_defined_for<T, T, Indices> and
    values::size_compares_with<collections::size_of<Indices>, index_count<T>, &stdcompat::is_gteq, applicability::permitted> and
    detail::index_collection_for_impl<Indices, T>::value;
#endif

}

#endif
