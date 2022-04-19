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

#ifndef OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
#define OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP

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
  template<fixed_coefficients Coeffs, std::invocable<const std::size_t> F>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    fixed_coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t>>>
#endif
  inline auto
  to_euclidean_coeff(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    return Coeffs::template to_euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Get a coordinate in Euclidean space corresponding to a coefficient in a matrix with typed coefficients.
   * \details This overload is operable for \ref dynamic_coefficients.
   * \tparam Coeffs The row coefficients for the transformed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix
   * and returning its scalar value.
   * \return The scalar value of the transformed coordinate in Euclidean space corresponding to the provided row.
   */
#ifdef __cpp_concepts
  template<dynamic_coefficients Coeffs, typename F> requires
    requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<dynamic_coefficients<Coeffs> and
    std::is_convertible_v<std::invoke_result_t<F&, std::size_t&>, const typename Coeffs::Scalar>>>
#endif
  inline auto
  to_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
  {
    return coeffs.to_euclidean_coeff(row, get_coeff);
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
  template<fixed_coefficients Coeffs, std::invocable<const std::size_t> F>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    fixed_coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t>>>
#endif
  inline auto
  from_euclidean_coeff(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    return Coeffs::template from_euclidean_array<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Get a typed coefficient corresponding to its corresponding coordinate in Euclidean space.
   * \details This overload is operable for \ref dynamic_coefficients.
   * \tparam Coeffs The row coefficients for the transformed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the transformed matrix.
   * \param get_coeff A function taking an index to a row (given some column) in the transformed matrix and
   * returning its scalar value.
   * \return The scalar value of the typed coefficient corresponding to the provided row.
   */
#ifdef __cpp_concepts
  template<dynamic_coefficients Coeffs, typename F> requires
    requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<dynamic_coefficients<Coeffs> and
    std::is_convertible_v<std::invoke_result_t<F&, std::size_t&>, const typename Coeffs::Scalar>>>
#endif
  inline auto
  from_euclidean_coeff(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
  {
    return coeffs.from_euclidean_coeff(row, get_coeff);
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
  template<fixed_coefficients Coeffs, std::invocable<const std::size_t> F>
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<
    fixed_coefficients<Coeffs> and std::is_invocable_v<F, const std::size_t>>>
#endif
  inline auto
  wrap_get(const std::size_t row, const F& get_coeff)
  {
    using Scalar = std::invoke_result_t<F, const std::size_t>;
    return Coeffs::template wrap_array_get<Scalar, 0>[row](get_coeff);
  }


  /**
   * \internal
   * \brief Wrap a given coefficient and return its wrapped, scalar value.
   * \details This overload is operable for \ref dynamic_coefficients.
   * \tparam Coeffs The row coefficients for the typed matrix.
   * \tparam F A function that takes a column index of a matrix and returns its unwrapped, scalar value.
   * \param row The applicable row of the matrix.
   * \return The scalar value of the wrapped coefficient corresponding to the provided
   * row and column (the column is an input into get_coeff).
   */
#ifdef __cpp_concepts
  template<dynamic_coefficients Coeffs, typename F> requires
    requires(F& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
  template<typename Coeffs, typename F, typename = std::enable_if_t<dynamic_coefficients<Coeffs> and
    std::is_convertible_v<std::invoke_result_t<F&, std::size_t&>, const typename Coeffs::Scalar>>>
#endif
  inline auto
  wrap_get(Coeffs&& coeffs, const std::size_t row, const F& get_coeff)
  {
    return coeffs.wrap_get(row, get_coeff);
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
  template<fixed_coefficients Coeffs, typename Scalar, typename FS, typename FG> requires
    requires(FS& f, std::size_t& i, Scalar& s) { f(i, s); } and
    requires(FG& f, std::size_t& i) { {f(i)} -> std::convertible_to<const Scalar>; }
#else
  template<typename Coeffs, typename Scalar, typename FS, typename FG, typename = std::enable_if_t<
    fixed_coefficients<Coeffs> and std::is_invocable_v<FS, std::size_t&, Scalar&> and
    std::is_convertible_v<std::invoke_result_t<FG&, std::size_t&>, const Scalar>>>
#endif
  inline void
  wrap_set(const std::size_t row, const Scalar s, const FS& set_coeff, const FG& get_coeff)
  {
    Coeffs::template wrap_array_set<Scalar, 0>[row](s, set_coeff, get_coeff);
  }


  /**
   * \internal
   * \brief Set the scalar value of a given typed coefficient in a matrix, and wrap the matrix column.
   * \details This overload is operable for \ref dynamic_coefficients.
   * \tparam Coeffs The row coefficients for the typed matrix.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam FS A function that takes an index and a Scalar value, and uses that value to set
   * a coefficient in a matrix, without any wrapping.
   * \tparam FG A function that takes an index to a column in a matrix and returns an unwrapped scalar
   * value corresponding to a matrix coefficient.
   * \param row The applicable row of the matrix.
   */
#ifdef __cpp_concepts
  template<dynamic_coefficients Coeffs, typename FS, typename FG> requires
    requires(FS& f, std::size_t& i, typename Coeffs::Scalar& s) { f(i, s); } and
    requires(FG& f, std::size_t& i) { {f(i)} -> std::convertible_to<const typename Coeffs::Scalar>; }
#else
  template<typename Coeffs, typename FS, typename FG, typename = std::enable_if_t<dynamic_coefficients<Coeffs> and
    std::is_invocable_v<FS, std::size_t&, typename Coeffs::Scalar&> and
    std::is_convertible_v<std::invoke_result_t<FG&, std::size_t&>, const typename Coeffs::Scalar>>>
#endif
  inline void
  wrap_set(Coeffs&& coeffs, const std::size_t row, const typename Coeffs::Scalar s,
           const FS& set_coeff, const FG& get_coeff)
  {
    coeffs.wrap_set(row, s, set_coeff, get_coeff);
  }


}// namespace OpenKalman::internal


#endif //OPENKALMAN_INDEX_DESCRIPTOR_FUNCTIONS_HPP
