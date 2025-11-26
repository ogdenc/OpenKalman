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
 * \internal
 * \brief Definition for \ref less_fixed_than.
 */

#ifndef OPENKALMAN_LESS_FIXED_THAN_HPP
#define OPENKALMAN_LESS_FIXED_THAN_HPP


namespace OpenKalman::internal
{
  namespace detail
  {
    template<typename T, std::size_t N, std::size_t...offset>
    constexpr bool an_extended_dim_is_dynamic_impl(std::index_sequence<offset...>)
    {
      return ((... or (dynamic_dimension<T, N + offset>)));
    }


    template<typename T, std::size_t N>
    constexpr bool an_extended_dim_is_dynamic()
    {
      if constexpr (index_count_v<T> != stdex::dynamic_extent and index_count_v<T> > N)
        return an_extended_dim_is_dynamic_impl<T, N>(std::make_index_sequence<index_count_v<T> - N>{});
      else
        return false;
    }


#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
    template<typename T, typename Descriptors, std::size_t...Ix>
    static constexpr bool less_fixed_than_impl(std::index_sequence<Ix...>)
    {
      return ((dynamic_dimension<T, Ix> and fixed_pattern<collections::collection_element_t<Ix, Descriptors>>) or ... or
        an_extended_dim_is_dynamic<T, sizeof...(Ix)>());
    }
#endif
  }


  /**
   * \brief \ref indexible T's vector space descriptors are less fixed than the at least one of the specified \ref vectors_space_descriptor_collection.
   */
  template<typename T, typename Descriptors>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept less_fixed_than =
    indexible<T> and pattern_collection<Descriptors> and
      (not pattern_collection<Descriptors> or
        []<std::size_t...Ix>(std::index_sequence<Ix...>)
          { return ((dynamic_dimension<T, Ix> and fixed_pattern<collections::collection_element_t<Ix, Descriptors>>) or ... or
              detail::an_extended_dim_is_dynamic<T, sizeof...(Ix)>()); }
          (std::make_index_sequence<collections::size_of_v<Descriptors>>{}));
#else
  constexpr bool less_fixed_than =
    indexible<T> and pattern_collection<Descriptors> and
    (not pattern_collection<Descriptors> or
      detail::less_fixed_than_impl<T, Descriptors>(std::make_index_sequence<collections::size_of_v<Descriptors>>{}));
#endif


}

#endif
