/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
  /**
   * \brief Take the trace of a matrix.
   * \details This is a generalized trace that applies to rectangular matrices. If the argument is rectangular,
   * this function returns the trace of the square sub-matrix.
   * \tparam Arg The matrix
   * \todo Redefine as a particular tensor contraction.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (max_tensor_order_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<(indexible<Arg>) and (max_tensor_order_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  trace(Arg&& arg)
  {
    if constexpr (zero<Arg> or empty_object<Arg>)
    {
      return internal::ScalarConstant<Qualification::unqualified, scalar_type_of_t<Arg>, 0>{};
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return internal::index_dimension_scalar_constant(arg, internal::smallest_dimension_index(arg));
    }
    else if constexpr (one_dimensional<Arg>)
    {
      return internal::get_singular_component(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      std::multiplies<scalar_type_of_t<Arg>> op;
      auto n = internal::index_dimension_scalar_constant(arg, internal::smallest_dimension_index(arg));
      return internal::scalar_constant_operation{op, constant_coefficient{arg}, n};
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      std::multiplies<scalar_type_of_t<Arg>> op;
      auto n = internal::index_dimension_scalar_constant(arg, internal::smallest_dimension_index(arg));
      return internal::scalar_constant_operation{op, constant_diagonal_coefficient{arg}, n};
    }
    else if constexpr (triangular_matrix<Arg>) // Includes the diagonal case.
    {
      return reduce(std::plus<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else // General case in which we have to add up the diagonal elements.
    {
      using Scalar = scalar_type_of_t<Arg>;
      auto diag = diagonal_of(std::forward<Arg>(arg));
      if constexpr(dynamic_dimension<decltype(diag), 0>)
      {
        auto dim = get_index_dimension_of<0>(diag);
        if (dim >= 2) return static_cast<Scalar>(reduce(std::plus<Scalar>{}, std::move(diag)));
        else if (dim == 1) return static_cast<Scalar>(internal::get_singular_component(std::move(diag)));
        else return static_cast<Scalar>(0); // d == 0 (empty vector)
      }
      else if constexpr (index_dimension_of_v<decltype(diag), 0> == 1)
      {
        return internal::get_singular_component(std::move(diag));
      }
      else
      {
        return reduce(std::plus<Scalar>{}, std::move(diag));
      }
    }
  }

} // namespace OpenKalman


#endif //OPENKALMAN_TRACE_HPP
