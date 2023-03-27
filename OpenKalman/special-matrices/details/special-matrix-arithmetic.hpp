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

  namespace
  {
    template<typename T>
    struct is_eigen_constant_expr : std::false_type {};

    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct is_eigen_constant_expr<ConstantAdapter<PatternMatrix, Scalar, constant...>> : std::true_type {};

    template<typename T>
  #ifdef __cpp_concepts
    concept eigen_constant_expr = is_eigen_constant_expr<std::decay_t<T>>::value;
  #else
    constexpr bool eigen_constant_expr = is_eigen_constant_expr<std::decay_t<T>>::value;
  #endif
  }


  // ---------- //
  //  negation  //
  // ---------- //

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
    if constexpr (zero_matrix<Arg>) return std::forward<Arg>(arg);
    else return make_constant_matrix_like(std::forward<Arg>(arg), internal::scalar_constant_operation<std::negate<>, constant_coefficient<Arg>>{});
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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(-nested_matrix(std::forward<Arg>(arg))));
  }


  // ---------- //
  //  addition  //
  // ---------- //

  /**
   * \brief indexible + indexible where at least one of them is zero
   */
#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2> requires
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(indexible<Arg1> and indexible<Arg2>) and
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
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
    maybe_has_same_shape_as<Arg1, Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    maybe_has_same_shape_as<Arg1, Arg2>, int> = 0>
#endif
  constexpr auto operator+(Arg1&& arg1, Arg2&& arg2)
  {
    using M = std::conditional_t<eigen_constant_expr<Arg1>, Arg1, Arg2>;
    using op = internal::scalar_constant_operation<std::plus<>, constant_coefficient<Arg1>, constant_coefficient<Arg2>>;
    if constexpr (not has_same_shape_as<Arg1, Arg2>) if (not get_index_descriptors_match(arg1, arg2))
      throw std::invalid_argument {"Arguments to operator+ have non-matching index descriptors"};
    return make_constant_matrix_like<M>(arg1, op{});
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
    if constexpr((lower_hermitian_adapter<Arg1> and lower_hermitian_adapter<Arg2>) or
      (upper_hermitian_adapter<Arg1> and upper_hermitian_adapter<Arg2>))
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = nested_matrix(std::forward<Arg1>(arg1)) + transpose(nested_matrix(std::forward<Arg2>(arg2)));
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
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
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = make_dense_writable_matrix_from(std::forward<Arg1>(arg1));
      set_triangle<triangle_type_of_v<Arg2>>(ret, ret + std::forward<Arg2>(arg2));
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
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) + nested_matrix(std::forward<Arg2>(arg2));
      return MatrixTraits<std::decay_t<Arg2>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
  }


  // ------------- //
  //  subtraction  //
  // ------------- //

  /**
   * \brief indexible - indexible where at least one of them is zero
   */
#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2> requires
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
    (dynamic_rows<Arg1> or dynamic_rows<Arg2> or row_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_columns<Arg2> or column_dimension_of_v<Arg1> == column_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(indexible<Arg1> and indexible<Arg2>) and
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
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
    maybe_has_same_shape_as<Arg1, Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    maybe_has_same_shape_as<Arg1, Arg2>, int> = 0>
#endif
  constexpr auto operator-(Arg1&& arg1, Arg2&& arg2)
  {
    using M = std::conditional_t<eigen_constant_expr<Arg1>, Arg1, Arg2>;
    using op = internal::scalar_constant_operation<std::minus<>, constant_coefficient<Arg1>, constant_coefficient<Arg2>>;
    if constexpr (not has_same_shape_as<Arg1, Arg2>) if (not get_index_descriptors_match(arg1, arg2))
      throw std::invalid_argument {"Arguments to operator- have non-matching index descriptors"};
    return make_constant_matrix_like<M>(arg1, op{});
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
    if constexpr((lower_hermitian_adapter<Arg1> and lower_hermitian_adapter<Arg2>) or
      (upper_hermitian_adapter<Arg1> and upper_hermitian_adapter<Arg2>))
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

    if constexpr (are_within_tolerance(constant, 0))
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
    if (s == 0) throw std::runtime_error("ConstantAdapber / 0: divide by zero error");

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
    return MatrixTraits<std::decay_t<Arg>>::make(make_self_contained<Arg>(nested_matrix(std::forward<Arg>(arg)) / scale));
  }


  // ----------------------- //
  //  matrix multiplication  //
  // ----------------------- //

  /**
   * \brief A matrix product, where one of the arguments is eigen_zero_expr
   */
#ifdef __cpp_concepts
  template<indexible Arg1, indexible Arg2> requires
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<indexible<Arg1> and indexible<Arg2> and
    ((eigen_constant_expr<Arg1> and zero_matrix<Arg1>) or (eigen_constant_expr<Arg2> and zero_matrix<Arg2>)) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    return make_zero_matrix_like<Arg1>(get_dimensions_of<0>(arg1), get_dimensions_of<1>(arg2));
  }


  /**
   * \brief constant_matrix * constant_matrix, where at least one of them is eigen_constant_expr
   */
#ifdef __cpp_concepts
  template<constant_matrix Arg1, constant_matrix Arg2> requires
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<constant_matrix<Arg1> and constant_matrix<Arg2> and
    (eigen_constant_expr<Arg1> or eigen_constant_expr<Arg2>) and
    (not zero_matrix<Arg1>) and (not zero_matrix<Arg2>) and
    (not identity_matrix<Arg1>) and (not identity_matrix<Arg2>) and
    (dynamic_columns<Arg1> or dynamic_rows<Arg2> or column_dimension_of<Arg1>::value == row_dimension_of<Arg2>::value), int> = 0>
#endif
  inline auto operator*(Arg1&& arg1, Arg2&& arg2)
  {
    return contract(std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
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
    return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(
      (nested_matrix(std::forward<Arg1>(arg1)).array() * nested_matrix(std::forward<Arg2>(arg2)).array()).matrix()));
  }


  /**
   * \brief (eigen_diagonal_expr * indexible) or (indexible * eigen_diagonal_expr)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((eigen_diagonal_expr<Arg1> and indexible<Arg2>) or (indexible<Arg1> and eigen_diagonal_expr<Arg2>)) and
    (column_dimension_of_v<Arg1> == row_dimension_of_v<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((eigen_diagonal_expr<Arg1> and indexible<Arg2>) or (indexible<Arg1> and eigen_diagonal_expr<Arg2>)) and
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
      return MatrixTraits<std::decay_t<Arg1>>::make(std::move(prod));
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
      return MatrixTraits<std::decay_t<Arg1>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
    }
    else
    {
      auto ret = std::forward<Arg1>(arg1) * std::forward<Arg2>(arg2).view();
      return MatrixTraits<std::decay_t<Arg2>>::make(make_self_contained<Arg1, Arg2>(std::move(ret)));
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
    return make_zero_matrix_like<Arg1>(get_dimensions_of<0>(arg1), get_dimensions_of<1>(arg2));
  }


  /*
   * ((self-adjoint or triangular) * native-eigen) or (native-eigen * (self-adjoint or triangular))
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
#endif
    (((eigen_self_adjoint_expr<Arg1> or eigen_triangular_expr<Arg1>) and not diagonal_matrix<Arg2>) or
     (not diagonal_matrix<Arg1> and (eigen_self_adjoint_expr<Arg2> or eigen_triangular_expr<Arg2>))) and
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


} // namespace OpenKalman

#endif //OPENKALMAN_SPECIAL_MATRIX_ARITHMETIC_HPP
