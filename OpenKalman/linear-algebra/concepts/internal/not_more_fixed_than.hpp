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
 * \brief Definition for \ref not_more_fixed_than.
 */

#ifndef OPENKALMAN_NOT_MORE_FIXED_THAN_HPP
#define OPENKALMAN_NOT_MORE_FIXED_THAN_HPP


namespace OpenKalman::internal
{
#if not defined(__cpp_concepts) or __cpp_generic_lambdas < 201707L
  namespace detail
  {
    template<typename T, typename Descriptors, std::size_t...Ix>
    static constexpr bool not_more_fixed_than_impl(std::index_sequence<Ix...>)
    {
      return (... and (dynamic_dimension<T, Ix> or fixed_pattern<collections::collection_element_t<Ix, Descriptors>>));
    }
  }
#endif


  /**
   * \brief \ref indexible T's vector space descriptors are not more fixed than the specified \ref vectors_space_descriptor_collection.
   */
  template<typename T, typename Descriptors>
#if defined(__cpp_concepts) and __cpp_generic_lambdas >= 201707L
  concept not_more_fixed_than =
    indexible<T> and pattern_collection<Descriptors> and
      (not pattern_tuple<Descriptors> or
        []<std::size_t...Ix>(std::index_sequence<Ix...>)
          { return (... and (dynamic_dimension<T, Ix> or fixed_pattern<collections::collection_element_t<Ix, Descriptors>>)); }
          (std::make_index_sequence<collections::size_of_v<Descriptors>>{}));
#else
  constexpr bool not_more_fixed_than =
    indexible<T> and pattern_collection<Descriptors> and
    (not pattern_tuple<Descriptors> or
      detail::not_more_fixed_than_impl<T, Descriptors>(std::make_index_sequence<collections::size_of_v<Descriptors>>{}));
#endif

}

#endif
