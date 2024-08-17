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
      if constexpr (index_count_v<T> != dynamic_size and index_count_v<T> > N)
        return an_extended_dim_is_dynamic_impl<T, N>(std::make_index_sequence<index_count_v<T> - N>{});
      else
        return false;
    }


    template<typename T, typename...Ds, std::size_t...IxD>
    static constexpr bool less_fixed_than_impl(std::index_sequence<IxD...>)
    {
      return ((dynamic_dimension<T, IxD> and fixed_vector_space_descriptor<Ds>) or ... or an_extended_dim_is_dynamic<T, sizeof...(IxD)>());
    }
  } // namespace detail


  /**
   * \brief \ref indexible T's vector space descriptors are less fixed than the set Ds, for at least one Ds.
   */
  template<typename T, typename...Ds>
#ifdef __cpp_concepts
  concept less_fixed_than =
#else
  constexpr bool less_fixed_than =
#endif
    indexible<T> and (vector_space_descriptor<Ds> and ...) and
      detail::less_fixed_than_impl<T, Ds...>(std::index_sequence_for<Ds...>{});

} // namespace OpenKalman::internal

#endif //OPENKALMAN_LESS_FIXED_THAN_HPP
