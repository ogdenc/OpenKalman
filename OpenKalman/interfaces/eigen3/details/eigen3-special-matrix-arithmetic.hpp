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

#include "eigen3-forward-declarations.hpp"

namespace OpenKalman::Eigen3
{
  // ---------- //
  //  negation  //
  // ---------- //

  /**
   * \brief negation of eigen_zero_expr
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
   * \brief negation of eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_constant_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    constexpr auto constant = MatrixTraits<Arg>::constant;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr std::size_t r = MatrixTraits<Arg>::rows;
    constexpr std::size_t c = MatrixTraits<Arg>::columns;

    if constexpr (constant == 0) return std::forward<Arg>(arg);
    else return ConstantMatrix<Scalar, -constant, r, c> {};
  }


  /**
   * \brief negation of eigen_diagonal_expr, eigen_self_adjoint_expr or eigen_triangular_expr
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
   * \brief eigen_matrix + eigen_matrix where one of them is zero_matrix
   */
#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns),
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
   * \brief eigen_constant_expr + eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  constexpr auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant + MatrixTraits<Arg2>::constant;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg1>::columns> {};
  }


  /**
   * \brief eigen_diagonal_expr + diagonal_matrix or diagonal_matrix + eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
  ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
    (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_self_adjoint_expr<Arg1> and eigen_self_adjoint_expr<Arg2> and
    (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((lower_triangular_storage<Arg1> and lower_triangular_storage<Arg2>) or
      (upper_triangular_storage<Arg1> and upper_triangular_storage<Arg2>))
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_triangular_expr<Arg1> and eigen_triangular_expr<Arg2> and
    not diagonal_matrix<Arg1> and not diagonal_matrix<Arg2> and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(OpenKalman::internal::same_triangle_type_as<Arg1, Arg2>)
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_native_matrix(std::forward<Arg1>(arg1));
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
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
      return to_diagonal(diagonal_of(std::forward<Arg1>(arg1)) + diagonal_of(std::forward<Arg2>(arg2)));
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
   * \brief eigen_matrix - eigen_matrix where one of them is zero_matrix
   */
#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns),
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
   * \brief eigen_constant_expr - eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  constexpr auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant - MatrixTraits<Arg2>::constant;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg1>::columns> {};
  }


  /**
   * \brief eigen_diagonal_expr - diagonal_matrix or diagonal_matrix - eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and diagonal_matrix<Arg2>) or
      (diagonal_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns), int> = 0>
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<eigen_self_adjoint_expr<Arg1> and eigen_self_adjoint_expr<Arg2> and
      (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
      (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr((lower_triangular_storage<Arg1> and lower_triangular_storage<Arg2>) or
      (upper_triangular_storage<Arg1> and upper_triangular_storage<Arg2>))
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<eigen_triangular_expr<Arg1> and eigen_triangular_expr<Arg2> and
      (not diagonal_matrix<Arg1>) and (not diagonal_matrix<Arg2>) and
      (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(OpenKalman::internal::same_triangle_type_as<Arg1, Arg2>)
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_native_matrix(std::forward<Arg1>(arg1));
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
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
      return to_diagonal(diagonal_of(std::forward<Arg1>(arg1)) - diagonal_of(std::forward<Arg2>(arg2)));
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
  template<eigen_zero_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  constexpr Arg&& operator*(Arg&& arg, const S scale)
  {
    return std::forward<Arg>(arg);
  }


  /**
   * \brief scalar * eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  constexpr Arg&& operator*(const S scale, Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  /**
   * \brief eigen_constant_expr * scalar
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg&& arg, const S scale)
  {
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (constant == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr std::size_t r = MatrixTraits<Arg>::rows;
      constexpr std::size_t c = MatrixTraits<Arg>::columns;
      return Eigen::Matrix<Scalar, r, c>::Constant(MatrixTraits<Arg>::constant * scale);
    }
  }


  /**
   * \brief scalar * eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  constexpr decltype(auto) operator*(const S scale, Arg&& arg)
  {
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (constant == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr std::size_t r = MatrixTraits<Arg>::rows;
      constexpr std::size_t c = MatrixTraits<Arg>::columns;
      return Eigen::Matrix<Scalar, r, c>::Constant(scale * MatrixTraits<Arg>::constant);
    }
  }


  /**
   * \brief (diagonal or self-adjoint or triangular) * scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator*(Arg&& arg, const S scale) noexcept
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) * scale));
  }


  /**
   * \brief scalar * (diagonal or self-adjoint or triangular)
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
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
  template<eigen_zero_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_zero_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
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
  template<eigen_constant_expr Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S>
#else
  template<typename Arg, typename S, std::enable_if_t<
    eigen_constant_expr<Arg> and std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  constexpr decltype(auto) operator/(Arg&& arg, const S s)
  {
    constexpr auto constant = MatrixTraits<Arg>::constant;

    if constexpr (constant == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr std::size_t r = MatrixTraits<Arg>::rows;
      constexpr std::size_t c = MatrixTraits<Arg>::columns;
      return Eigen::Matrix<Scalar, r, c>::Constant(MatrixTraits<Arg>::constant / s);
    }
  }


  /**
   * \brief (eigen_diagonal_expr or self-adjoint_matrix or triangular_matrix) / scalar
   */
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<typename MatrixTraits<Arg>::Scalar> S> requires
    eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>
#else
  template<typename Arg, typename S, std::enable_if_t<
    (eigen_diagonal_expr<Arg> or eigen_self_adjoint_expr<Arg> or eigen_triangular_expr<Arg>) and
    std::is_convertible_v<S, typename MatrixTraits<Arg>::Scalar>, int> = 0>
#endif
  inline auto operator/(Arg&& arg, const S scale) noexcept
  {
    return MatrixTraits<Arg>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) / scale));
  }


  // ----------------------- //
  //  matrix multiplication  //
  // ----------------------- //

  /**
   * \brief eigen_matrix * eigen_matrix where one of them is zero_matrix
   */
#ifdef __cpp_concepts
  template<eigen_matrix Arg1, eigen_matrix Arg2> requires (zero_matrix<Arg1> or zero_matrix<Arg2>) and
  (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_matrix<Arg1> and eigen_matrix<Arg2> and (zero_matrix<Arg1> or zero_matrix<Arg2>) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::rows;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    if constexpr (dynamic_rows<Arg1> and dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, 0, 0> {row_count(arg1), column_count(arg2)};
    }
    else if constexpr (dynamic_rows<Arg1> and not dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, rows, cols> {row_count(arg1)};
    }
    else if constexpr (not dynamic_rows<Arg1> and dynamic_columns<Arg2>)
    {
      return ZeroMatrix<Scalar, rows, cols> {column_count(arg2)};
    }
    else
    {
      static_assert(not dynamic_rows<Arg1> and not dynamic_columns<Arg2>);
      return ZeroMatrix<Scalar, rows, cols> {};
    }
  }


  /**
   * \brief eigen_constant_expr * eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<eigen_constant_expr Arg1, eigen_constant_expr Arg2> requires
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows) and
    std::same_as<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<eigen_constant_expr<Arg1> and eigen_constant_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows) and
    std::is_same_v<typename MatrixTraits<Arg1>::Scalar, typename MatrixTraits<Arg2>::Scalar>, int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    constexpr auto newconst = MatrixTraits<Arg1>::constant * MatrixTraits<Arg2>::constant * MatrixTraits<Arg2>::rows;
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    return ConstantMatrix<Scalar, newconst, MatrixTraits<Arg1>::rows, MatrixTraits<Arg2>::columns> {};
  }


/**
 * \brief eigen_constant_expr * identity_matrix or identity_matrix * eigen_constant_expr
 */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_constant_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and eigen_constant_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_constant_expr<Arg1> and identity_matrix<Arg2>) or
      (identity_matrix<Arg1> and eigen_constant_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
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
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    eigen_diagonal_expr<Arg1> and eigen_diagonal_expr<Arg2> and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and (not identity_matrix<Arg1>) and
    (not identity_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    return MatrixTraits<Arg1>::make(make_self_contained<Arg1, Arg2>(
      (nested_matrix(std::forward<Arg1>(arg1)).array() * nested_matrix(std::forward<Arg2>(arg2)).array()).matrix()));
  }


  /**
   * \brief eigen_diagonal_expr * eigen_matrix or eigen_matrix * eigen_diagonal_expr
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_diagonal_expr<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(zero_matrix<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(identity_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(identity_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(eigen_constant_expr<Arg1>)
    {
      return (MatrixTraits<Arg1>::constant *
        transpose(native_matrix(std::forward<Arg2>(arg2)))).template replicate<MatrixTraits<Arg2>::rows, 1>();
    }
    else if constexpr(eigen_constant_expr<Arg2>)
    {
      return (native_matrix(std::forward<Arg1>(arg1)) *
        MatrixTraits<Arg2>::constant).template replicate<1, MatrixTraits<Arg2>::columns>();
    }
    else if constexpr(eigen_diagonal_expr<Arg1>)
    {
      // This operation result will never be self-contained because Eigen::DiagonalWrapper does not store its vector.
      return nested_matrix(std::forward<Arg1>(arg1)).asDiagonal() * std::forward<Arg2>(arg2);
    }
    else
    {
      // See comment above.
      return std::forward<Arg1>(arg1) * nested_matrix(std::forward<Arg2>(arg2)).asDiagonal();
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    auto prod = make_self_contained(std::forward<Arg1>(arg1).view() * make_native_matrix(std::forward<Arg2>(arg2)));
    static_assert(eigen_matrix<decltype(prod)> or eigen_diagonal_expr<decltype(prod)>);
    if constexpr(OpenKalman::internal::same_triangle_type_as<Arg1, Arg2>)
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
    (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
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
    and (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
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
    (MatrixTraits<Arg1>::rows == MatrixTraits<Arg2>::rows)
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
    (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::rows;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    return ZeroMatrix<Scalar, rows, cols> {};
  }


  /*
   * ((self-adjoint or triangular) * native-eigen) or (native-eigen * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and (not zero_matrix<Arg1>) and
    (not zero_matrix<Arg2>) and (MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::rows)
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
