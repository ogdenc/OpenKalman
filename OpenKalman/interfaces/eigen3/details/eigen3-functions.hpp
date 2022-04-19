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
 * \brief Overloaded functions relating to various Eigen3 types
 */

#ifndef OPENKALMAN_EIGEN3_FUNCTIONS_HPP
#define OPENKALMAN_EIGEN3_FUNCTIONS_HPP


namespace OpenKalman
{
  /**
   * Make a native Eigen matrix from a list of coefficients in row-major order.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, std::size_t rows, std::size_t columns = 1, std::convertible_to<Scalar> ... Args>
  requires
    (rows == dynamic_size and columns == dynamic_size) or
    (rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
    (columns == dynamic_size and sizeof...(Args) % rows == 0) or
    (rows == dynamic_size and sizeof...(Args) % columns == 0)
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    scalar_type<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    ((rows == dynamic_size and columns == dynamic_size) or
    (rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
    (columns == dynamic_size and sizeof...(Args) % rows == 0) or
    (rows == dynamic_size and sizeof...(Args) % columns == 0)), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args ... args)
  {
    using M = Eigen3::eigen_matrix_t<Scalar, rows, columns>;
    return MatrixTraits<M>::make(static_cast<const Scalar>(args)...);
  }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif


  /**
   * \overload
   * \brief Make a native Eigen matrix from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, scalar_type ... Args>
  requires
    (rows == dynamic_size and columns == dynamic_size) or
    ((rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
    (columns == dynamic_size and sizeof...(Args) % rows == 0) or
    (rows == dynamic_size and sizeof...(Args) % columns == 0))
#else
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    (scalar_type<Args> and ...) and
    ((rows == dynamic_size and columns == dynamic_size) or
    ((rows != dynamic_size and columns != dynamic_size and sizeof...(Args) == rows * columns) or
    (columns == dynamic_size and sizeof...(Args) % rows == 0) or
    (rows == dynamic_size and sizeof...(Args) % columns == 0))), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_eigen_matrix<Scalar, rows, columns>(args...);
  }
#ifndef __cpp_concepts
# pragma GCC diagnostic pop
#endif


  /// Make a native Eigen 1-column vector from a list of coefficients in row-major order.
#ifdef __cpp_concepts
  template<scalar_type ... Args>
#else
  template<typename ... Args, std::enable_if_t<(scalar_type<Args> and ...), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_eigen_matrix<Scalar, sizeof...(Args), 1>(args...);
  }

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_FUNCTIONS_HPP
