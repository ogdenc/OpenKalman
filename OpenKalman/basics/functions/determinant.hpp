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
  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> Arg> requires (max_tensor_order_of_v<Arg> <= 2)
  constexpr std::convertible_to<scalar_type_of_t<Arg>> auto
#else
  template<typename Arg, std::enable_if_t<square_matrix<Arg, Likelihood::maybe> and (max_tensor_order_of_v<Arg> <= 2), int> = 0>
  constexpr auto
#endif
  determinant(Arg&& arg)
  {
    constexpr auto ix = []{ if constexpr (dynamic_dimension<Arg, 0>) return 1; else return 0; }();

    if constexpr (identity_matrix<Arg>)
    {
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 1>{};
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return internal::constexpr_pow(constant_diagonal_coefficient{arg}, internal::index_dimension_scalar_constant_of<ix>(arg))();
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 1> or dimension_size_of_index_is<Arg, 1, 1>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return constant_coefficient {arg};
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{};
    }
    else if constexpr (dimension_size_of_index_is<Arg, 0, 0> or dimension_size_of_index_is<Arg, 1, 0>)
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_is_square(arg))
        throw std::domain_error {"Argument to 'determinant' is not a square matrix"};
      return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 1>{};
    }
    else if constexpr (triangular_matrix<Arg> and not dynamic_dimension<Arg, ix> and index_dimension_of_v<Arg, ix> >= 2) // this includes the diagonal case
    {
      return reduce(std::multiplies<scalar_type_of_t<Arg>>{}, diagonal_of(std::forward<Arg>(arg)));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (has_dynamic_dimensions<Arg>)
      {
        auto d = get_is_square(arg);
        if (not d) throw std::invalid_argument{"Argument of 'determinant' is not a square matrix."};
        else if (*d >= 2) return static_cast<scalar_type_of_t<Arg>>(0);
        else if (*d == 1) return static_cast<scalar_type_of_t<Arg>>(constant_coefficient {arg});
        else return static_cast<scalar_type_of_t<Arg>>(1); // empty matrix
      }
      else
      {
        return internal::ScalarConstant<Likelihood::definitely, scalar_type_of_t<Arg>, 0>{};
      }
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_DETERMINANT_HPP
