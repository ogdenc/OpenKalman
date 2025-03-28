/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref vector_space_descriptors_may_match_with.
 */

#ifndef OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MAY_MATCH_WITH_HPP
#define OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MAY_MATCH_WITH_HPP

#include "linear-algebra/coordinates/concepts/compares_with.hpp"

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t I, typename T, typename...Ts>
    constexpr bool maybe_equivalent_descriptors_per_index()
    {
      return (... and compares_with<vector_space_descriptor_of_t<T, I>, vector_space_descriptor_of_t<Ts, I>, equal_to<>, Applicability::permitted>);
    }

    template<typename...Ts, std::size_t...Is>
    constexpr bool vector_space_descriptors_may_match_with_impl(std::index_sequence<Is...>)
    {
      if constexpr (sizeof...(Ts) < 2) return true;
      else return (... and maybe_equivalent_descriptors_per_index<Is, Ts...>());
    }
  } // namespace detail

  /**
   * \brief Specifies that \ref indexible objects Ts may have equivalent dimensions and vector-space types.
   * \details Two dimensions are considered the same if their \ref coordinate::pattern are equivalent.
   * \sa vector_space_descriptors_match_with
   * \sa vector_space_descriptors_match
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept vector_space_descriptors_may_match_with =
#else
  constexpr bool vector_space_descriptors_may_match_with =
#endif
    (indexible<Ts> and ...) and
    detail::vector_space_descriptors_may_match_with_impl<Ts...>(std::make_index_sequence<std::max({std::size_t{0}, index_count_v<Ts>...})>{});


} // namespace OpenKalman

#endif //OPENKALMAN_VECTOR_SPACE_DESCRIPTORS_MAY_MATCH_WITH_HPP
