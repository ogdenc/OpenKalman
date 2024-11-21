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
   * \tparam D0 The \ref vector_space_descriptor for the rows.
   * \tparam D1 The \ref vector_space_descriptor for the columns.
   */
#ifdef __cpp_concepts
  template<indexible Arg, vector_space_descriptor D0, vector_space_descriptor D1> requires
    (not static_vector_space_descriptor<D0> or not static_vector_space_descriptor<D1> or internal::prefix_of<D0, D1> or internal::prefix_of<D1, D0>) and
    (dynamic_dimension<Arg, 0> or maybe_equivalent_to<vector_space_descriptor_of<Arg, 0>, D0> or maybe_equivalent_to<vector_space_descriptor_of<Arg, 0>, D1>)
  constexpr diagonal_matrix auto
#else
  template<typename Arg, typename D0, typename D1, std::enable_if_t<
    indexible<Arg> and vector_space_descriptor<D0> and vector_space_descriptor<D1> and
      (not static_vector_space_descriptor<D0> or not static_vector_space_descriptor<D1> or internal::prefix_of<D0, D1> or internal::prefix_of<D1, D0>) and
      (dynamic_dimension<Arg, 0> or maybe_equivalent_to<vector_space_descriptor_of<Arg, 0>, D0> or maybe_equivalent_to<vector_space_descriptor_of<Arg, 0>, D1>), int> = 0>
  constexpr auto
#endif
  make_diagonal_adapter(Arg&& arg, D0&& d0, D1&& d1)
  {
    return DiagonalAdapter {std::forward<Arg>(arg), std::forward<D0>(d0), std::forward<D1>(d1)};
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
    auto d = get_vector_space_descriptor<0>(arg);
    return DiagonalAdapter {std::forward<Arg>(arg), d, d};
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_DIAGONAL_ADAPTER_HPP
