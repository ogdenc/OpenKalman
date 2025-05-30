/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \make_eigen_matrix function
 */

#ifndef OPENKALMAN_MAKE_EIGEN_MATRIX_HPP
#define OPENKALMAN_MAKE_EIGEN_MATRIX_HPP


namespace OpenKalman::Eigen3
{
  /**
   * Make a native Eigen matrix from a list of coefficients in row-major order.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
#ifdef __cpp_concepts
  template<values::number Scalar, std::size_t rows, std::size_t columns = 1, std::convertible_to<Scalar> ... Args>
    requires (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns)
#else
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    values::number<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns), int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args...args)
  {
    using M = Eigen3::eigen_matrix_t<Scalar, rows, columns>;
    std::tuple d_tup {Dimensions<rows>{}, Dimensions<columns>{}};
    return make_dense_object_from<M>(std::move(d_tup), static_cast<const Scalar>(args)...);
  }


  /**
   * \overload
   * \brief In this overload, the scalar type is derived from the arguments.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, values::number...Args> requires
    (rows != dynamic_size) and (columns != dynamic_size) and (sizeof...(Args) == rows * columns) and
    requires { requires values::number<std::common_type_t<Args...>>; }
#else
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<(values::number<Args> and ...) and
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
  template<values::number Scalar, std::convertible_to<Scalar>...Args> requires (not std::is_void_v<Scalar>)
#else
  template<typename Scalar, typename ... Args, std::enable_if_t<
    values::number<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and (not std::is_void_v<Scalar>), int> = 0>
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
  template<typename Scalar = void, values::number ... Args> requires (std::is_void_v<Scalar>) and
    requires { requires values::number<std::common_type_t<Args...>>; }
#else
  template<typename Scalar = void, typename ... Args, std::enable_if_t<(values::number<Args> and ...) and std::is_void_v<Scalar>, int> = 0>
#endif
  inline auto
  make_eigen_matrix(const Args...args)
  {
    return make_eigen_matrix<std::common_type_t<Args...>>(args...);
  }


} // namespace OpenKalman::Eigen3


#endif //OPENKALMAN_MAKE_EIGEN_MATRIX_HPP
