/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref to_diagonal function.
 */

#ifndef OPENKALMAN_TO_DIAGONAL_HPP
#define OPENKALMAN_TO_DIAGONAL_HPP

namespace OpenKalman
{
  using namespace interface;

  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<dimension_size_of_index_is<1, 1, Likelihood::maybe> Arg>
#else
  template<typename Arg, std::enable_if_t<dimension_size_of_index_is<Arg, 1, 1, Likelihood::maybe>, int> = 0>
#endif
  constexpr decltype(auto)
  to_diagonal(Arg&& arg)
  {
    constexpr auto dim = row_dimension_of_v<Arg>;

    if constexpr (dim == 1)
    {
      if constexpr (diagonal_matrix<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else
      {
        if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
          throw std::domain_error {"Argument of to_diagonal must be a column vector, not a row vector"};
        return DiagonalMatrix {std::forward<Arg>(arg)};
      }
    }
    else if constexpr (zero_matrix<Arg> and dim != dynamic_size)
    {
      if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
        throw std::domain_error {"Argument of to_diagonal must have 1 column; instead it has " +
          std::to_string(get_index_dimension_of<1>(arg))};
      return make_zero_matrix_like<Arg>(Dimensions<dim>{}, Dimensions<dim>{});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_DIAGONAL_HPP
