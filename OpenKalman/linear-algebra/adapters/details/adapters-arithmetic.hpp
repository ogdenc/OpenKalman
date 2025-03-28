/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Arithmetic definitions for Eigen extensions
 * \todo remove file
 */

#ifndef OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP
#define OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP

#if false
namespace OpenKalman
{
  /**
   * \brief Addition involving ConstantAdapter, DiagonalAdapter, TriangularAdapter, or HermitianAdapter.
   */
#ifdef __cpp_concepts
  template<indexible Arg1, vector_space_descriptors_may_match_with Arg2> requires
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1>) or
    (constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<vector_space_descriptors_may_match_with<Arg1, Arg2> and
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1>) or
    (constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    return sum(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
  }


  /**
   * \brief Subtraction involving ConstantAdapter, DiagonalAdapter, TriangularAdapter, or HermitianAdapter.
   */
#ifdef __cpp_concepts
  template<indexible Arg1, vector_space_descriptors_may_match_with Arg2> requires
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1>) or
    (constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<vector_space_descriptors_may_match_with<Arg1, Arg2> and
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1>) or
    (constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>), int> = 0>
#endif
  constexpr auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    return sum(std::forward<Arg1>(arg1), -std::forward<Arg2>(arg2));
  }


  /**
   * \brief Matrix multiplication involving ConstantAdapter, DiagonalAdapter, TriangularAdapter, or HermitianAdapter.
   */
#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2> requires
    (dynamic_dimension<Arg1, 1> or dynamic_dimension<Arg2, 0> or compares_with<vector_space_descriptor_of_t<Arg1, 1>, vector_space_descriptor_of_t<Arg2, 0>>) and
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1> or
    constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (dynamic_dimension<Arg1, 1> or dynamic_dimension<Arg2, 0> or compares_with<typename vector_space_descriptor_of<Arg1, 1>::type, typename vector_space_descriptor_of<Arg2, 0>::type>) and
    (constant_adapter<Arg1> or internal::diagonal_expr<Arg1> or internal::triangular_expr<Arg1> or internal::hermitian_expr<Arg1> or
    constant_adapter<Arg2> or internal::diagonal_expr<Arg2> or internal::triangular_expr<Arg2> or internal::hermitian_expr<Arg2>), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    return contract(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
  }

} // namespace OpenKalman
#endif

#endif //OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP
