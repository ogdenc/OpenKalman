/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief The trace function.
 */

#ifndef OPENKALMAN_TRACE_HPP
#define OPENKALMAN_TRACE_HPP

namespace OpenKalman
{
#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   * \todo Redefine as a particular tensor contraction.
   */
  template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (max_tensor_order_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<(square_shaped<Arg, Qualification::depends_on_dynamic_shape>) and (max_tensor_order_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  trace(Arg&& arg)
  {
    constexpr std::size_t ix = []{ if constexpr (dynamic_dimension<Arg, 0>) return 1; else return 0; }();

    if constexpr (identity_matrix<Arg>)
    {
      return internal::index_dimension_scalar_constant<ix>(arg);
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      std::multiplies<scalar_type_of_t<Arg>> op;
      return internal::scalar_constant_operation{op, constant_diagonal_coefficient{arg}, internal::index_dimension_scalar_constant<ix>(arg)};
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> or dimension_size_of_index_is<Arg, 1, 1>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      return constant_coefficient {arg};
    }
    else if constexpr (zero<Arg> or dimension_size_of_index_is<Arg, 0, 0> or dimension_size_of_index_is<Arg, 1, 0>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      return internal::ScalarConstant<Qualification::unqualified, scalar_type_of_t<Arg>, 0>{};
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg))
        throw std::domain_error {"Argument to 'trace' is not a square matrix"};
      std::multiplies<scalar_type_of_t<Arg>> op;
      return internal::scalar_constant_operation{op, constant_coefficient{arg}, internal::index_dimension_scalar_constant<ix>(arg)};
    }
    else if constexpr (triangular_matrix<Arg> and not dynamic_dimension<Arg, ix> and index_dimension_of_v<Arg, ix> >= 2) // this includes the diagonal case
    {
      return reduce(std::plus<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else
    {
      auto diag = diagonal_of(std::forward<Arg>(arg));
      if constexpr(dynamic_dimension<decltype(diag), 0>)
      {
        auto dim = get_index_dimension_of<0>(diag);
        if (dim >= 2) return static_cast<scalar_type_of_t<Arg>>(reduce(std::plus<scalar_type_of_t<Arg>>{}, std::move(diag)));
        else if (dim == 1) return static_cast<scalar_type_of_t<Arg>>(constant_coefficient {std::move(diag)});
        else return static_cast<scalar_type_of_t<Arg>>(0);
      }
      else
      {
        // diag is known at compile time to have at least 2 dimensions
        return reduce(std::plus<scalar_type_of_t<Arg>>{}, std::move(diag));
      }
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_TRACE_HPP
