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
 * \brief Arithmetic definitions for Eigen3 extensions
 */

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP

#include "interfaces/eigen3/details/eigen3-forward-declarations.hpp"

namespace OpenKalman::Eigen3
{
  // ---------- //
  //  negation  //
  // ---------- //

  /**
   * \brief Negation of /ref eigen_zero_expr.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr Arg&& operator-(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  /**
   * \brief Negation of /ref eigen_constant_expr.
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    constexpr auto constant = constant_coefficient_v<Arg>;
    return make_constant_matrix_like<-constant>(std::forward<Arg>(arg));
  }


  /**
   * \brief Negation of /ref eigen_diagonal_expr, /ref eigen_self_adjoint_expr or /ref eigen_triangular_expr
   */
#ifdef __cpp_concepts
  template<typename Arg> requires eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto operator-(Arg&& arg)
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(-nested_matrix(std::forward<Arg>(arg))));
  }


  // ---------- //
  //  addition  //
  // ---------- //

  /**
   * \brief eigen_matrix + eigen_matrix where at least one of them is a eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value),
    int> = 0>
#endif
  constexpr auto&& operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      static_assert(zero_matrix<Arg1>);
      return std::forward<Arg2>(arg2);
    }
  }


  /**
   * \brief constant_matrix + constant_matrix, where at least one of them is eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2> requires
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>) and
    std::same_as<scalar_type_of_t<Arg1>, scalar_type_of_t<Arg2>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value) and
    std::is_same_v<typename scalar_type_of<Arg1>::type, typename scalar_type_of<Arg2>::type>, int> = 0>
#endif
  constexpr auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = constant_coefficient_v<Arg1> + constant_coefficient_v<Arg2>;
    return make_constant_matrix_like<newconst>(std::forward<Arg1>(arg1));
  }


  /**
   * \brief eigen_diagonal_expr + diagonal_matrix or diagonal_matrix + eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
  ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
    (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (zero_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else
    {
      return to_diagonal(make_self_contained<Arg1, Arg2>(
        diagonal_of(std::forward<Arg1>(arg1)) + diagonal_of(std::forward<Arg2>(arg2))));
    }
  }


  /**
   * \brief self-adjoint + self-adjoint
   */
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg1, eigen_self_adjoint_expr Arg2> requires
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_self_adjoint_expr<Arg1> and eigen_self_adjoint_expr<Arg2> and
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((lower_self_adjoint_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
      (upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>))
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + transpose(nested_matrix(std::forward<Arg2>(arg2)));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
  }


  /**
   * \brief triangular + triangular
   */
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg1, eigen_triangular_expr Arg2> requires
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_triangular_expr<Arg1> and eigen_triangular_expr<Arg2> and
    not diagonal_matrix<Arg1> and not diagonal_matrix<Arg2> and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(triangle_type_of_v<Arg1> == triangle_type_of_v<Arg2>)
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_dense_writable_matrix_from(std::forward<Arg1>(arg1));
      constexpr auto mode = upper_triangular_matrix<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() += nested_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  /*
   * ((self-adjoint or triangular) + diagonal) or (diagonal + (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(zero_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (diagonal_matrix<Arg1> and diagonal_matrix<Arg2>)
    {
      return make_self_contained(
        to_diagonal(diagonal_of(std::forward<Arg1>(arg1)) + diagonal_of(std::forward<Arg2>(arg2))));
    }
    else if constexpr (eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>)
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + std::forward<Arg2>(arg2);
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg2>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
  }


  // ------------- //
  //  subtraction  //
  // ------------- //

  /**
   * \brief eigen_matrix - eigen_matrix where at least one of them is eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value),
    int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      static_assert(zero_matrix<Arg1>);
      return -std::forward<Arg2>(arg2);
    }
  }


  /**
   * \brief constant_matrix - constant_matrix where at least one of them is eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2> requires
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>) and
    std::same_as<scalar_type_of_t<Arg1>, scalar_type_of_t<Arg2>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value) and
    std::is_same<typename scalar_type_of<Arg1>::type, typename scalar_type_of<Arg2>::type>::value, int> = 0>
#endif
  constexpr auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = constant_coefficient_v<Arg1> - constant_coefficient_v<Arg2>;
    return make_constant_matrix_like<newconst>(std::forward<Arg1>(arg1));
  }


  /**
   * \brief eigen_diagonal_expr - diagonal_matrix or diagonal_matrix - eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    (column_dimension_of<Arg1>::value == column_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (zero_matrix<Arg1>)
    {
      return -std::forward<Arg2>(arg2);
    }
    else
    {
      return to_diagonal(make_self_contained<Arg1, Arg2>(
        diagonal_of(std::forward<Arg1>(arg1)) - diagonal_of(std::forward<Arg2>(arg2))));
    }
  }


  /**
   * \brief self-adjoint - self-adjoint
   */
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg1, eigen_self_adjoint_expr Arg2> requires
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<eigen_self_adjoint_expr<Arg1> and eigen_self_adjoint_expr<Arg2> and
      (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
      (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((lower_self_adjoint_matrix<Arg1> and lower_self_adjoint_matrix<Arg2>) or
      (upper_self_adjoint_matrix<Arg1> and upper_self_adjoint_matrix<Arg2>))
    {
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(
        nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2))));
    }
    else
    {
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(
        nested_matrix(std::forward<Arg1>(arg1)) - transpose(nested_matrix(std::forward<Arg2>(arg2)))));
    }
  }


  /**
   * \brief triangular - triangular
   */
#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg1, eigen_triangular_expr Arg2> requires
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<eigen_triangular_expr<Arg1> and eigen_triangular_expr<Arg2> and
      (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
      (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(triangle_type_of_v<Arg1> == triangle_type_of_v<Arg2>)
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_dense_writable_matrix_from(std::forward<Arg1>(arg1));
      constexpr auto mode = upper_triangular_matrix<Arg2> ? Eigen::Upper : Eigen::Lower;
      ret.template triangularView<mode>() -= nested_matrix(std::forward<Arg2>(arg2));
      return ret;
    }
  }


  /**
   * \brief ((self-adjoint or triangular) - diagonal) or (diagonal - (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(zero_matrix<Arg1>)
    {
      return -std::forward<Arg2>(arg2);
    }
    else if constexpr (diagonal_matrix<Arg1> and diagonal_matrix<Arg2>)
    {
      return make_self_contained(
        to_diagonal(diagonal_of(std::forward<Arg1>(arg1)) - diagonal_of(std::forward<Arg2>(arg2))));
    }
    else if constexpr(eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>)
    {
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(
        nested_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2)));
    }
    else
    {
      return MatrixTraits<Arg2>::make(make_self_contained<Arg1, Arg2>(
        std::forward<Arg1>(arg1) - nested_matrix(std::forward<Arg2>(arg2))));
    }
  }


  // ----------------------- //
  //  scalar multiplication  //
  // ----------------------- //

  /**
   * \brief eigen_zero_expr * scalar
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr Arg&& operator*(Arg&& arg, const S scale)
  {
    return std::forward<Arg>(arg);
  }


  /**
   * \brief scalar * eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr Arg&& operator*(const S scale, Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  /**
   * \brief eigen_constant_expr * scalar
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg&& arg, const S scale)
  {
    constexpr auto constant = constant_coefficient_v<Arg>;

    if constexpr (constant == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return to_native_matrix<pattern_matrix_of_t<Arg>>(std::forward<Arg>(arg)) * scale;
    }
  }


  /**
   * \brief scalar * eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr decltype(auto) operator*(const S scale, Arg&& arg)
  {
    return operator*(std::forward<Arg>(arg), scale);
  }


  /**
   * \brief (diagonal or self-adjoint or triangular) * scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<scalar_type_of_t<Arg>> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto operator*(Arg&& arg, const S scale) noexcept
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) * scale));
  }


  /**
   * \brief scalar * (diagonal or self-adjoint or triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<scalar_type_of_t<Arg>> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto operator*(const S scale, Arg&& arg) noexcept
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(scale * nested_matrix(std::forward<Arg>(arg))));
  }


  // ----------------- //
  //  scalar division  //
  // ----------------- //

  /**
   * \brief Divide an \ref eigen_zero_expr by a scalar.
   * \tparam Arg1 An \ref eigen_zero_expr.
   * \tparam Arg2 An arithmetic scalar type.
   * \return If it does not throw a divide-by-zero exception, the result will be \ref eigen_zero_expr.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr Arg&& operator/(Arg&& arg, const S s)
  {
    if (s == 0) throw std::runtime_error("ZeroMatrix / 0: divide by zero error");
    return std::forward<Arg>(arg);
  }


  /**
   * \brief eigen_constant_expr / scalar
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg, std::convertible_to<scalar_type_of_t<Arg>> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  constexpr decltype(auto) operator/(Arg&& arg, const S s)
  {
    if (s == 0) throw std::runtime_error("ConstantMatrix / 0: divide by zero error");

    if constexpr (zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return to_native_matrix<pattern_matrix_of_t<Arg>>(std::forward<Arg>(arg)) / s;
    }
  }


  /**
   * \brief (eigen_diagonal_expr or self-adjoint_matrix or triangular_matrix) / scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<scalar_type_of_t<Arg>> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale) noexcept
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) / scale));
  }


  // ----------------------- //
  //  matrix multiplication  //
  // ----------------------- //

  /**
   * \brief A matrix product, where one of the arguments is eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
  (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(eigen_zero_expr<Arg1> or eigen_zero_expr<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    constexpr auto rows = index_dimension_of_v<Arg1, 0>;
    constexpr auto cols = index_dimension_of_v<Arg2, 1>;

    if constexpr (dynamic_rows<Arg1> and dynamic_columns<Arg2>)
      return make_zero_matrix_like<Arg1, rows, cols>(runtime_dimension_of<0>(arg1), runtime_dimension_of<1>(arg2));
    else if constexpr (dynamic_rows<Arg1> and not dynamic_columns<Arg2>)
      return make_zero_matrix_like<Arg1, rows, cols>(runtime_dimension_of<0>(arg1));
    else if constexpr (not dynamic_rows<Arg1> and dynamic_columns<Arg2>)
      return make_zero_matrix_like<Arg1, rows, cols>(runtime_dimension_of<1>(arg2));
    else
      return make_zero_matrix_like<Arg1, rows, cols>();
  }


  /**
   * \brief constant_matrix * constant_matrix, where at least one of them is eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2> requires
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    std::same_as<scalar_type_of_t<Arg1>, scalar_type_of_t<Arg2>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value) and
    std::is_same_v<typename scalar_type_of<Arg1>::type, typename scalar_type_of<Arg2>::type>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (dynamic_columns<Arg1> or dynamic_rows<Arg2>) assert (runtime_dimension_of<1>(arg1) == runtime_dimension_of<0>(arg2));

    constexpr auto newconst = constant_coefficient_v<Arg1> * constant_coefficient_v<Arg2> * row_dimension_of_v<Arg2>;
    return make_constant_matrix_like<Arg1, newconst>(get_dimensions_of<0>(arg1), get_dimensions_of<1>(arg2));
  }


/**
 * \brief eigen_constant_expr * identity_matrix or identity_matrix * eigen_constant_expr
 */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_constant_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and eigen_constant_expr<Arg2>)) and
    (column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_constant_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and eigen_constant_expr<Arg2>)) and
    (column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto&& operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_constant_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  /**
   * \brief eigen_diagonal_expr * eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg1, eigen_diagonal_expr Arg2> requires
   (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_diagonal_expr<Arg1> and eigen_diagonal_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(
      (nested_matrix(std::forward<Arg1>(arg1)).array() * nested_matrix(std::forward<Arg2>(arg2)).array()).matrix()));
  }


  /**
   * \brief (eigen_diagonal_expr * eigen_matrix) or (eigen_matrix * eigen_diagonal_expr)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_diagonal_expr<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (zero_matrix<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (identity_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (identity_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (constant_matrix<Arg2>)
    {
      auto column0 = diagonal_of(std::forward<Arg1>(arg1)) * constant_coefficient_v<Arg2>;
      return make_self_contained((std::move(column0)).template replicate<1, column_dimension_of_v<Arg2>>());
    }
    else if constexpr(constant_matrix<Arg1>)
    {
      auto row0 = transpose(diagonal_of(std::forward<Arg2>(arg2))) * constant_coefficient_v<Arg1>;
      return make_self_contained((std::move(row0)).template replicate<row_dimension_of_v<Arg2>, 1>());
    }
    else if constexpr (diagonal_matrix<Arg1> and diagonal_matrix<Arg2>)
    {
      return make_self_contained<Arg1, Arg2>(
        to_diagonal(diagonal_of(std::forward<Arg1>(arg1)) + diagonal_of(std::forward<Arg2>(arg2))));
    }
    else if constexpr (eigen_diagonal_expr<Arg1>)
    {
      return make_self_contained<Arg1, Arg2>(
        diagonal_of(std::forward<Arg1>(arg1)).asDiagonal() * std::forward<Arg2>(arg2));
    }
    else
    {
      static_assert(eigen_diagonal_expr<Arg2>);
      return make_self_contained<Arg1, Arg2>(
        std::forward<Arg1>(arg1) * diagonal_of(std::forward<Arg2>(arg2)).asDiagonal());
    }
  }


  /**
   * \brief (self-adjoint or triangular) * (self-adjoint or triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and
      (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>)) and
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    auto prod = make_dense_writable_matrix_from(std::forward<Arg2>(arg2));
    prod.applyOnTheLeft(std::forward<Arg1>(arg1).view());

    if constexpr(triangle_type_of_v<Arg1> == triangle_type_of_v<Arg2>)
    {
      return MatrixTraits<Arg1>::make(std::move(prod));
    }
    else
    {
      return prod;
    }
  }


  /**
   * \brief (self-adjoint * diagonal) or (diagonal * self-adjoint)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((eigen_self_adjoint_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_self_adjoint_expr<Arg2>)) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_self_adjoint_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1).view() * std::forward<Arg2>(arg2);
    }
    else
    {
      return std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).view();
    }
  }


  /*
   * (triangular * diagonal) or (diagonal * triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    ((eigen_triangular_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_triangular_expr<Arg2>)) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>)
    and (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_triangular_expr<Arg1>)
    {
      auto ret = std::forward<Arg1>(arg1).view() * std::forward<Arg2>(arg2);
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).view();
      return MatrixTraits<Arg2>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
  }


  /*
   * ((self-adjoint or triangular) * identity) or (identity * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (row_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  /*
   * ((self-adjoint or triangular) * zero) or (zero * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and zero_matrix<Arg2>) or
      (zero_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto rows = index_dimension_of_v<Arg1, 0>;
    constexpr auto cols = index_dimension_of_v<Arg2, 1>;

    return make_zero_matrix_like<Arg1, rows, cols>();
  }


  /*
   * ((self-adjoint or triangular) * native-eigen) or (native-eigen * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and
        eigen_matrix<Arg2> and not diagonal_matrix<Arg2>) or
     (eigen_matrix<Arg1> and not diagonal_matrix<Arg1> and
        (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1).view() * std::forward<Arg2>(arg2);
    }
    else
    {
      return std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).view();
    }
  }


} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_ARITHMETIC_HPP
