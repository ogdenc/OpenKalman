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
 * \brief Definitions for \ref make_diagonal_adapter.
 */

#ifndef OPENKALMAN_MAKE_DIAGONAL_ADAPTER_HPP
#define OPENKALMAN_MAKE_DIAGONAL_ADAPTER_HPP

namespace OpenKalman
{
  /**
   * \brief Make a \ref diagonal_matrix, specifying the first two dimensions, which may not necessarily be the same.
   * \tparam Arg A vector or higher-order tensor reflecting the diagonal(s).
   * \tparam D0 The \ref coordinates::pattern for the rows.
   * \tparam D1 The \ref coordinates::pattern for the columns.
   */
#ifdef __cpp_concepts
  template<indexible Arg, coordinates::pattern D0, coordinates::pattern D1> requires
    (not fixed_pattern<D0> or not fixed_pattern<D1> or coordinates::compares_with<D0, D1, less_equal<>> or coordinates::compares_with<D1, D0, less_equal<>>) and
    (dynamic_dimension<Arg, 0> or compares_with<vector_space_descriptor_of<Arg, 0>, D0, equal_to<>, applicability::permitted> or compares_with<vector_space_descriptor_of<Arg, 0>, D1, equal_to<>, applicability::permitted>)
  constexpr diagonal_matrix auto
#else
  template<typename Arg, typename D0, typename D1, std::enable_if_t<
    indexible<Arg> and coordinates::pattern<D0> and coordinates::pattern<D1> and
      (not fixed_pattern<D0> or not fixed_pattern<D1> or coordinates::compares_with<D0, D1, less_equal<>> or coordinates::compares_with<D1, D0, less_equal<>>) and
      (dynamic_dimension<Arg, 0> or compares_with<vector_space_descriptor_of<Arg, 0>, D0, equal_to<>, applicability::permitted> or compares_with<vector_space_descriptor_of<Arg, 0>, D1, equal_to<>, applicability::permitted>), int> = 0>
  constexpr auto
#endif
  make_diagonal_adapter(Arg&& arg, D0&& d0, D1&& d1)
  {
    return diagonal_adapter {std::forward<Arg>(arg), std::forward<D0>(d0), std::forward<D1>(d1)};
  }


  /**
   * \overload
   * \brief Make an square \ref diagonal_matrix that is square with respect to dimensions 0 and 1.
   * \tparam Arg A vector or higher-order tensor reflecting the diagonal(s).
   */
#ifdef __cpp_concepts
  template<indexible Arg>
  constexpr diagonal_matrix auto
#else
  template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
  constexpr auto
#endif
  make_diagonal_adapter(Arg&& arg)
  {
    auto d = get_pattern_collection<0>(arg);
    return diagonal_adapter {std::forward<Arg>(arg), d, d};
  }

}

#endif
