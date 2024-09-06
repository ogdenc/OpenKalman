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
 * \brief The determinant function.
 */

#ifndef OPENKALMAN_DETERMINANT_HPP
#define OPENKALMAN_DETERMINANT_HPP


namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg>
    inline void error_if_argument_to_determinant_is_not_square(const Arg& arg)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not is_square_shaped(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
    }
  } // namespace detail


  /**
   * \brief Take the determinant of a matrix.
   * \tparam Arg A square matrix
   */
#ifdef __cpp_concepts
  template<square_shaped<Qualification::depends_on_dynamic_shape> Arg> requires (max_tensor_order_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<square_shaped<Arg, Qualification::depends_on_dynamic_shape> and (max_tensor_order_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  determinant(Arg&& arg)
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto ix = []{ if constexpr (dynamic_dimension<Arg, 0>) return 1; else return 0; }();

    if constexpr (identity_matrix<Arg> or empty_object<Arg>)
    {
      detail::error_if_argument_to_determinant_is_not_square(arg);
      return values::ScalarConstant<Scalar, 1>{};
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> or dimension_size_of_index_is<Arg, 1, 1>)
    {
      // At least one of the dimensions is 1.
      detail::error_if_argument_to_determinant_is_not_square(arg);
      return internal::get_singular_component(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg> and not dynamic_dimension<Arg, ix> and index_dimension_of_v<Arg, ix> >= 2)
    {
      detail::error_if_argument_to_determinant_is_not_square(arg);
      return values::ScalarConstant<Scalar, 0>{};
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      detail::error_if_argument_to_determinant_is_not_square(arg);
      return internal::constexpr_pow(constant_diagonal_coefficient{arg}, internal::index_to_scalar_constant<Scalar>(get_index_dimension_of<ix>(arg)))();
    }
    else if constexpr (triangular_matrix<Arg>) // Includes the diagonal case.
    {
      detail::error_if_argument_to_determinant_is_not_square(arg);
      return reduce(std::multiplies<Scalar>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (constant_matrix<Arg>) // Arg could be empty or 1D at runtime, so we need to check.
    {
      auto d = is_square_shaped(arg);
      if (not d) throw std::invalid_argument{"Argument of 'determinant' is not a square matrix."};
      else if (get_dimension_size_of(*d) >= 2) return static_cast<Scalar>(0);
      else if (get_dimension_size_of(*d) == 1) return static_cast<Scalar>(internal::get_singular_component(std::forward<Arg>(arg))); // 1D matrix
      else return static_cast<Scalar>(1); // empty matrix
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_DETERMINANT_HPP
