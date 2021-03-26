/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions for accessing elements of typed arrays, based on typed coefficients.
 */

#ifndef OPENKALMAN_COEFFICIENT_FUNCTIONS_HPP
#define OPENKALMAN_COEFFICIENT_FUNCTIONS_HPP

#include <type_traits>
#include <functional>

#ifdef __cpp_concepts
#include <concepts>
#endif

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Get a coordinate in Euclidean space corresponding to a coefficient in a matrix with typed coefficients.
   * \tparam Coeffs The row coefficients for the transformed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix
   * and returning its scalar value.
   * \return The scalar value of the transformed coordinate in Euclidean space corresponding to the provided row.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, std::invocable<const std::size_t> F>
  requires std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t> and
    std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>>>
#endif
  inline std::invoke_result_t<F, const std::size_t>
  to_euclidean_coeff(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    return Coeffs::template to_euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
   * \tparam Coeffs The row coefficients for the transformed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
   * returning its scalar value.
   * \return The scalar value of the typed coefficient corresponding to the provided row.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, std::invocable<const std::size_t> F>
  requires std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t> and
    std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>>>
#endif
  inline std::invoke_result_t<F, const std::size_t>
  from_euclidean_coeff(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    return Coeffs::template from_euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Wrap a given coefficient and return its wrapped, scalar value.
   * \tparam Coeffs The row coefficients for the typed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the matrix.
   * \return The scalar value of the wrapped coefficient corresponding to the provided
   * row and column (the column is an input into get_coeff).
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, std::invocable<const std::size_t> F>
    requires std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t> and
    std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>>>
#endif
  inline std::invoke_result_t<F, const std::size_t>
  wrap_get(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    static_assert(std::is_arithmetic_v<Scalar>);
    return Coeffs::template wrap_array_get<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
   * \tparam Coeffs The row coefficients for the typed matrix.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam FS A function that takes an index and a Scalar value, and uses that value to set
   * a coefficient in a matrix, without any wrapping.
   * \tparam FG A function that takes an index to a column in a matrix and returns an unwrapped scalar
   * value corresponding to a matrix coefficient.
   * \param row The applicable row of the matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, typename Scalar,
    std::invocable<const std::size_t, const Scalar> FS, std::invocable<const std::size_t> FG>
    requires std::is_arithmetic_v<Scalar> and std::is_arithmetic_v<std::invoke_result_t<FG, const std::size_t>>
#else
  template<typename Coeffs, typename Scalar, typename FS, typename FG, typename = std::enable_if_t<
    coefficients<Coeffs> and std::is_arithmetic_v<Scalar> and std::is_invocable_v<FG, const std::size_t> and
    std::is_invocable_v<FS, const std::size_t, const Scalar> and
    std::is_arithmetic_v<std::invoke_result_t<FG, const std::size_t>>>>
#endif
  inline void
  wrap_set(const std::size_t row, const Scalar s, const FS& set_coeff, const FG& get_coeff)
  {
    Coeffs::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
  }


}// namespace OpenKalman::internal


#endif //OPENKALMAN_COEFFICIENT_FUNCTIONS_HPP
