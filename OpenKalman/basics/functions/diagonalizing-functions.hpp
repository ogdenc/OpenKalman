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
 * \brief Functions for converting to and from a diagonal matrix or tensor.
 */

#ifndef OPENKALMAN_DIAGONALIZING_FUNCTIONS_HPP
#define OPENKALMAN_DIAGONALIZING_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // ============= //
  //  to_diagonal  //
  // ============= //

  /**
   * \brief Convert a column vector into a diagonal matrix.
   * \tparam Arg A column vector matrix
   * \returns A diagonal matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires dimension_size_of_index_is<Arg, 1, 1, Likelihood::maybe>
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
        return to_native_matrix(std::forward<Arg>(arg));
      }
      else
      {
        if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
          throw std::domain_error {"Argument of to_diagonal must be a column vector, not a row vector"};
        return DiagonalMatrix {std::forward<Arg>(arg)};
      }
    }
    else if constexpr (constant_matrix<Arg> or (zero_matrix<Arg> and dynamic_rows<Arg>))
    {
      return DiagonalMatrix {std::forward<Arg>(arg)};
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_columns<Arg>) if (get_index_dimension_of<1>(arg) != 1)
        throw std::domain_error {"Argument of to_diagonal must have 1 column; instead it has " +
          std::to_string(get_index_dimension_of<1>(arg))};
      return make_zero_matrix_like<Arg>(Dimensions<dim>{}, Dimensions<dim>{});
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::to_diagonal(std::forward<Arg>(arg));
    }
  }


  // ============= //
  //  diagonal_of  //
  // ============= //

  namespace detail
  {
    template<typename Arg>
    constexpr void check_if_square_at_runtime(const Arg& arg)
    {
      if constexpr (not square_matrix<Arg>)
      if (get_dimensions_of<0>(arg) != get_dimensions_of<1>(arg))
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
    auto dim = get_dimensions_of<dynamic_rows<Arg> ? 1 : 0>(arg);

    if constexpr (one_by_one_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{});
    }
    else if constexpr (zero_matrix<Arg>)
    {
      detail::check_if_square_at_runtime(arg);
      return make_zero_matrix_like<Arg>(dim, Dimensions<1>{});
    }
    else if constexpr (constant_matrix<Arg> or constant_diagonal_matrix<Arg>)
    {
      detail::check_if_square_at_runtime(arg);
      constexpr auto c = []{
        if constexpr (constant_matrix<Arg>) return constant_coefficient_v<Arg>;
        else return constant_diagonal_coefficient_v<Arg>;
      }();

#  if __cpp_nontype_template_args >= 201911L
      return make_constant_matrix_like<Arg, c>(dim, Dimensions<1>{});
#  else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<scalar_type_of_t<Arg>>(c_integral)))
        return make_constant_matrix_like<Arg, c_integral>(dim, Dimensions<1>{});
      else
        return make_self_contained(c * make_constant_matrix_like<Arg, 1>(dim, Dimensions<1>{}));
#  endif
    }
    else
    {
      return interface::Conversions<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONALIZING_FUNCTIONS_HPP
