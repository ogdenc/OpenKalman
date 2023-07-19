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
    requires (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns)
#else
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    scalar_type<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args...args)
  {
    using M = Eigen3::eigen_matrix_t<Scalar, rows, columns>;
    std::tuple d_tup {Dimensions<rows>{}, Dimensions<columns>{}};
    return make_dense_writable_matrix_from<M>(std::move(d_tup), static_cast<const Scalar>(args)...);
  }


  /**
   * \overload
   * \brief In this overload, the scalar type is derived from the arguments.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, scalar_type...Args> requires
    (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns) and
    requires { scalar_type<std::common_type_t<Args...>>; }
#else
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and
    (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args...args)
  {
    return make_eigen_matrix<std::common_type_t<Args...>, rows, columns>(args...);
  }


  /**
   * \overload
   * \brief In this overload, the result is a column vector of size determined by the number of arguments.
   */
#ifdef __cpp_concepts
  template<scalar_type Scalar, std::convertible_to<Scalar>...Args> requires (not std::is_void_v<Scalar>)
#else
  template<typename Scalar, typename ... Args, std::enable_if_t<
    scalar_type<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and (not std::is_void_v<Scalar>), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args ... args)
  {
    return make_eigen_matrix<Scalar, sizeof...(Args), 1>(args...);
  }


  /**
   * \overload
   * \brief In this overload, the scalar type is derived from the arguments.
   */
#ifdef __cpp_concepts
  template<typename Scalar = void, scalar_type ... Args> requires (std::is_void_v<Scalar>) and
    requires { scalar_type<std::common_type_t<Args...>>; }
#else
  template<typename Scalar = void, typename ... Args, std::enable_if_t<(scalar_type<Args> and ...) and std::is_void_v<Scalar>, int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args...args)
  {
    return make_eigen_matrix<std::common_type_t<Args...>>(args...);
  }

} // namespace OpenKalman


#endif //OPENKALMAN_EIGEN3_FUNCTIONS_HPP
