/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COEFFICIENT_FUNCTIONS_H
#define OPENKALMAN_COEFFICIENT_FUNCTIONS_H

#include <type_traits>
#include <array>
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
   * \tparam Scalar The scalar type of the transformed matrix.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a column in the transformed matrix and returning its Scalar value.
   * \return The Scalar value of the transformed coordinate in Euclidean space corresponding to the provided
   * row and column (the column is an input into get_coeff).
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
  template<typename Coeffs, typename Scalar, std::enable_if_t<coefficients<Coeffs>, int> = 0>
#endif
  static Scalar to_Euclidean(const std::size_t row, const std::function<Scalar(const std::size_t)> get_coeff)
  {
    return Coeffs::template to_Euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
   * \tparam Coeffs The row coefficients for the transformed matrix.
   * \tparam Scalar The scalar type of the transformed matrix.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a column in the transformed matrix and returning its Scalar value.
   * \return The Scalar value of the typed coefficient corresponding to the provided
   * row and column (the column is an input into get_coeff).
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, typename Scalar> requires std::is_arithmetic_v<Scalar>
#else
  template<typename Coeffs, typename Scalar, std::enable_if_t<coefficients<Coeffs>, int> = 0>
#endif
  static Scalar from_Euclidean(const std::size_t row, const std::function<Scalar(const std::size_t)> get_coeff)
  {
    return Coeffs::template from_Euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Wrap a given coefficient and return its wrapped, scalar value.
   * \tparam Coeffs The row coefficients for the typed matrix.
   * \tparam F An invocable function that takes a column index of a matrix and returns its unwrapped, Scalar value.
   * \param row The applicable row of the matrix.
   * \return The Scalar value of the wrapped coefficient corresponding to the provided
   * row and column (the column is an input into get_coeff).
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, std::invocable<const std::size_t> F>
    requires std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>
#else
  template<typename Coeffs, typename F, std::enable_if_t<
    coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t> and
    std::is_arithmetic_v<std::invoke_result_t<F, const std::size_t>>, int> = 0>
#endif
  static auto wrap_get(const std::size_t row, const F& get_coeff)
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
   * \tparam FS An invocable function that takes an index and a Scalar value, and uses that value to set
   * a coefficient in a matrix, without any wrapping.
   * \tparam FG An invocable function that takes an index to a column in a matrix and returns an unwrapped scalar
   * value corresponding to a matrix coefficient.
   * \param row The applicable row of the matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coeffs, typename Scalar,
    std::invocable<const std::size_t, const Scalar> FS, std::invocable<const std::size_t> FG>
    requires std::is_arithmetic_v<Scalar> and std::is_arithmetic_v<std::invoke_result_t<FG, const std::size_t>>
#else
  template<typename Coeffs, typename Scalar, typename FS, typename FG, std::enable_if_t<
    coefficients<Coeffs> and std::is_arithmetic_v<Scalar> and std::is_invocable_v<FG, const std::size_t> and
    std::is_invocable_v<FS, const std::size_t, const Scalar> and
    std::is_arithmetic_v<std::invoke_result_t<FG, const std::size_t>>, int> = 0>
#endif
  static void wrap_set(const Scalar s, const std::size_t row, const FS& set_coeff, const FG& get_coeff)
  {
    Coeffs::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
  }


}// namespace OpenKalman::internal


#endif //OPENKALMAN_COEFFICIENT_FUNCTIONS_H
