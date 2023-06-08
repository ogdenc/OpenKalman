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

#ifndef OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP
#define OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP

namespace OpenKalman
{
  // ---------- //
  //  negation  //
  // ---------- //


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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(-nested_matrix(std::forward<Arg>(arg))));
  }


  // ---------- //
  //  addition  //
  // ---------- //

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
      return to_diagonal(sum(diagonal_of(std::forward<Arg1>(arg1)), diagonal_of(std::forward<Arg2>(arg2))));
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
    if constexpr((hermitian_adapter<Arg1, HermitianAdapterType::lower> and hermitian_adapter<Arg2, HermitianAdapterType::lower>) or
      (hermitian_adapter<Arg1, HermitianAdapterType::upper> and hermitian_adapter<Arg2, HermitianAdapterType::upper>))
    {
      return MatrixTraits<std::decay_t<Arg1>>::make(sum(nested_matrix(std::forward<Arg1>(arg1)), nested_matrix(std::forward<Arg2>(arg2))));
    }
    else
    {
      return MatrixTraits<std::decay_t<Arg1>>::make(sum(nested_matrix(std::forward<Arg1>(arg1)), transpose(nested_matrix(std::forward<Arg2>(arg2)))));
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
      auto ret = sum(nested_matrix(std::forward<Arg1>(arg1)), nested_matrix(std::forward<Arg2>(arg2)));
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_dense_writable_matrix_from(std::forward<Arg1>(arg1));
      set_triangle<triangle_type_of_v<Arg2>>(ret, sum(ret, std::forward<Arg2>(arg2)));
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
      return sum(to_diagonal(diagonal_of(std::forward<Arg1>(arg1)), diagonal_of(std::forward<Arg2>(arg2))));
    }
    else if constexpr (eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>)
    {
      return MatrixTraits<std::decay_t<Arg1>>::make(sum(nested_matrix(std::forward<Arg1>(arg1)), std::forward<Arg2>(arg2)));
    }
    else
    {
      return MatrixTraits<std::decay_t<Arg2>>::make(sum(std::forward<Arg1>(arg1), nested_matrix(std::forward<Arg2>(arg2))));
    }
  }


  // ------------- //
  //  subtraction  //
  // ------------- //

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
    if constexpr((hermitian_adapter<Arg1, HermitianAdapterType::lower> and hermitian_adapter<Arg2, HermitianAdapterType::lower>) or
      (hermitian_adapter<Arg1, HermitianAdapterType::upper> and hermitian_adapter<Arg2, HermitianAdapterType::upper>))
    {
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(
        nested_matrix(std::forward<Arg1>(arg1)) - nested_matrix(std::forward<Arg2>(arg2))));
    }
    else
    {
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(
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
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_dense_writable_matrix_from(std::forward<Arg1>(arg1));
      set_triangle<triangle_type_of_v<Arg2>>(ret, ret - std::forward<Arg2>(arg2));
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
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(
        nested_matrix(std::forward<Arg1>(arg1)) - std::forward<Arg2>(arg2)));
    }
    else
    {
      return MatrixTraits<std::decay_t<Arg2>>::make(make_self_contained<Arg1, Arg2>(
        std::forward<Arg1>(arg1) - nested_matrix(std::forward<Arg2>(arg2))));
    }
  }


  // ----------------------- //
  //  scalar multiplication  //
  // ----------------------- //

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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) * scale));
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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(scale * nested_matrix(std::forward<Arg>(arg))));
  }


  // ----------------- //
  //  scalar division  //
  // ----------------- //

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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) / scale));
  }


  // ----------------------- //
  //  matrix multiplication  //
  // ----------------------- //

  /**
   * \brief (special matrix * indexible) or (indexible * special matrix)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (not constant_adapter<Arg1>) and (not constant_adapter<Arg2>) and
    ((eigen_diagonal_expr<Arg1> or eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) or
      (eigen_diagonal_expr<Arg2> or eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>)) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(not constant_adapter<Arg1>) and (not constant_adapter<Arg2>) and
    ((eigen_diagonal_expr<Arg1> or eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) or
      (eigen_diagonal_expr<Arg2> or eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>)) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    return contract(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP
