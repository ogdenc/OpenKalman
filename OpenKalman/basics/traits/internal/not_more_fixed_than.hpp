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
  namespace detail
  {
    template<typename T, typename...Ds, std::size_t...IxD>
    static constexpr bool not_more_fixed_than_impl(std::index_sequence<IxD...>)
    {
      return (... and (dynamic_dimension<T, IxD> or fixed_vector_space_descriptor<Ds>));
    }
  } // namespace detail


  /**
   * \brief \ref indexible T's vector space descriptors are not more fixed than the set Ds for any of Ds.
   */
  template<typename T, typename...Ds>
#ifdef __cpp_concepts
  concept not_more_fixed_than =
#else
  constexpr bool not_more_fixed_than =
#endif
    indexible<T> and (vector_space_descriptor<Ds> and ...) and
      detail::not_more_fixed_than_impl<T, Ds...>(std::index_sequence_for<Ds...>{});

} // namespace OpenKalman::internal

#endif //OPENKALMAN_NOT_MORE_FIXED_THAN_HPP
