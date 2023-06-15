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
 * \brief Definition for \ref diagonal_of function.
 */

#ifndef OPENKALMAN_DIAGONAL_OF_HPP
#define OPENKALMAN_DIAGONAL_OF_HPP

namespace OpenKalman
{
  using namespace interface;

  namespace detail
  {
    template<typename Arg>
    constexpr void check_if_square_at_runtime(const Arg& arg)
    {
      if constexpr (not square_matrix<Arg>) if (get_index_descriptor<0>(arg) != get_index_descriptor<1>(arg))
        throw std::invalid_argument {"Argument of diagonal_of must be a square matrix; instead, " +
        (get_index_dimension_of<0>(arg) == get_index_dimension_of<1>(arg) ?
          "the row and column indices have non-equivalent types" :
          "it has " + std::to_string(get_index_dimension_of<0>(arg)) + " rows and " +
            std::to_string(get_index_dimension_of<1>(arg)) + " columns")};
    };
  }


  /**
   * \brief Extract the diagonal from a square matrix.
   * \tparam Arg A diagonal matrix
   * \returns Arg A column vector
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg>
  constexpr /*dimension_size_of_index_is<1, 1>*/ decltype(auto)
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe>, int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    auto dim = get_index_descriptor<dynamic_rows<Arg> ? 1 : 0>(arg);

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<Arg, scalar_type_of_t<Arg>, 1>(dim, Dimensions<1>{});
    }
    else if constexpr (zero_matrix<Arg>)
    {
      detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<Arg>(dim, Dimensions<1>{});
    }
    else if constexpr (constant_matrix<Arg>)
    {
      detail::check_if_square_at_runtime(arg);
      return make_constant_matrix_like<Arg>(constant_coefficient{arg}, dim, Dimensions<1>{});
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      detail::check_if_square_at_runtime(arg);
      return make_constant_matrix_like<Arg>(constant_diagonal_coefficient{arg}, dim, Dimensions<1>{});
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
