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

#ifndef OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP

#include <type_traits>
#include <random>

namespace OpenKalman::Eigen3
{
  /**
   * Make a native Eigen matrix from a list of coefficients in row-major order.
   * \tparam Scalar The scalar type of the matrix.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, std::convertible_to<Scalar> ... Args>
    requires std::is_arithmetic_v<Scalar> and (sizeof...(Args) == rows * columns)
#else
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    std::is_arithmetic_v<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    (sizeof...(Args) == rows * columns), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using M = Eigen::Matrix<Scalar, rows, columns>;
    return MatrixTraits<M>::make(static_cast<const Scalar>(args)...);
  }


  /**
   * \overload
   * \brief Make a native Eigen matrix from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, typename ... Args> requires
    (std::is_arithmetic_v<Args> and ...) and (sizeof...(Args) == rows * columns)
#else
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    (std::is_arithmetic_v<Args> and ...) and (sizeof...(Args) == rows * columns), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_native_matrix<Scalar, rows, columns>(args...);
  }


  /// Make a native Eigen 1-column vector from a list of coefficients in row-major order.
#ifdef __cpp_concepts
  template<typename ... Args> requires(std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_native_matrix<Scalar, sizeof...(Args), 1>(args...);
  }


  /**
   * Convert to a self-contained Eigen3 matrix.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires
    eigen_matrix<Arg> or
    eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or
    eigen_diagonal_expr<Arg>
#else
  template<typename Arg, std::enable_if_t<
    eigen_matrix<Arg> or
    eigen_self_adjoint_expr<Arg> or
    eigen_triangular_expr<Arg> or
    eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_native_matrix(Arg&& arg) noexcept
  {
    if constexpr (std::is_same_v<std::decay_t<Arg>, native_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return native_matrix_t<Arg> {std::forward<Arg>(arg)};
    }
  }


  /// Convert to function-returnable version of the matrix.
#ifdef __cpp_concepts
  template<typename ... Ts, eigen_native Arg>
#else
  template<typename ... Ts, typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<Arg> or ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return make_native_matrix(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr std::size_t row_count(Arg&& arg)
  {
    if constexpr (dynamic_rows<Arg>)
      return arg.rows();
    else
      return MatrixTraits<Arg>::rows;
  }


#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr std::size_t column_count(Arg&& arg)
  {
    if constexpr (dynamic_columns<Arg>)
      return arg.cols();
    else
      return MatrixTraits<Arg>::columns;
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, eigen_matrix Arg> requires (Coefficients::dimensions == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and eigen_matrix<Arg> and
    (Coefficients::dimensions == MatrixTraits<Arg>::rows), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg) noexcept
  {
    if constexpr (Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return ToEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, eigen_matrix Arg> requires
    (Coefficients::euclidean_dimensions == MatrixTraits<Arg>::rows)
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<coefficients<Coefficients> and eigen_matrix<Arg> and
    (Coefficients::euclidean_dimensions == MatrixTraits<Arg>::rows), int> = 0>
#endif
  constexpr decltype(auto)
  from_euclidean(Arg&& arg) noexcept
  {
    if constexpr (Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return FromEuclideanExpr<Coefficients, Arg>(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<coefficients Coefficients, eigen_matrix Arg>
#else
  template<typename Coefficients, typename Arg, std::enable_if_t<
    coefficients<Coefficients> and eigen_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    if constexpr (Coefficients::axes_only or identity_matrix<Arg> or zero_matrix<Arg>)
    {
      /// \todo: Add functionality to conditionally wrap zero and identity, depending on wrap min and max.
      return std::forward<Arg>(arg);
    }
    else
    {
      return from_euclidean<Coefficients>(to_euclidean<Coefficients>(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<one_by_one_matrix Arg>
#else
  template<typename Arg, std::enable_if_t<one_by_one_matrix<Arg>, int> = 0>
#endif
  constexpr Arg&&
  to_diagonal(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg> requires column_vector<Arg> and (not one_by_one_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    column_vector<Arg> and (not one_by_one_matrix<Arg>), int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      if constexpr (dynamic_rows<Arg>)
      {
        const std::size_t dim = row_count(arg);
        return ZeroMatrix<Scalar, 0, 0>(dim, dim);
      }
      else
      {
        constexpr auto dim = MatrixTraits<Arg>::rows;
        return ZeroMatrix<Scalar, dim, dim>();
      }
    }
    else
    {
      return DiagonalMatrix<Arg> {std::forward<Arg>(arg)};
    }
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg> requires one_by_one_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and one_by_one_matrix<Arg>, int> = 0>
#endif
  constexpr Arg&&
  diagonal_of(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and
    (dynamic_shape<Arg> or (square_matrix<Arg> and not one_by_one_matrix<Arg>)), int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    if constexpr (identity_matrix<Arg>)
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      if constexpr (dynamic_rows<Arg>)
      {
        using Mat = Eigen::Matrix<Scalar, Eigen::internal::traits<std::decay_t<Arg>>::RowsAtCompileTime, 1>;
        assert(row_count(arg) == column_count(arg));
        return Mat::Constant(row_count(arg), 1, 1);
      }
      else
      {
        constexpr std::size_t dim = MatrixTraits<Arg>::rows;
        return ConstantMatrix<Scalar, 1, dim, 1> {};
      }
    }
    else
    {
      return std::forward<Arg>(arg).diagonal();
    }
  }


  /**
   * \brief Take the transpose of an eigen native matrix
   * \note The result has the same lifetime as arg.
   */
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return std::forward<Arg>(arg).transpose();
    }
  }


  /**
   * \brief Take the adjoint of an eigen native matrix
   * \note The result has the same lifetime as arg.
   */
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  inline decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    if constexpr (complex_number<typename MatrixTraits<Arg>::Scalar>)
    {
      return std::forward<Arg>(arg).adjoint();
    }
    else
    {
      return transpose(std::forward<Arg>(arg));
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(1);
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return arg(0, 0);
    }
    else
    {
      return arg.determinant();
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and square_matrix<Arg>, int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
    }
    else if constexpr (identity_matrix<Arg>)
    {
      return row_count(arg);
    }
    else if constexpr (one_by_one_matrix<Arg>)
    {
      return arg(0, 0);
    }
    else
    {
      return arg.trace();
    }
  }


  /// Solve the equation AX = B for X. A is an invertible square matrix. (Does not check that A is invertible.)
  /// Uses the square LU decomposition.
#ifdef __cpp_concepts
  template<eigen_native A, eigen_matrix B> requires square_matrix<A> and
    (MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<eigen_native<A> and eigen_matrix<B> and square_matrix<A> and
    (MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  constexpr auto
  solve(const A& a, const B& b)
  {
    using M = eigen_matrix_t<typename MatrixTraits<B>::Scalar, MatrixTraits<A>::rows, MatrixTraits<B>::columns>;
    if constexpr (one_by_one_matrix<A>)
    {
      return M(b(0, 0)/a(0, 0));
    }
    else
    {
      return M(a.lu().solve(b));
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return std::forward<Arg>(arg).rowwise().sum() / MatrixTraits<Arg>::columns;
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_native A>
#else
  template<typename A, std::enable_if_t<eigen_native<A>, int> = 0>
#endif
  constexpr auto
  LQ_decomposition(A&& a)
  {
    if constexpr (diagonal_matrix<A> or lower_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto dim = MatrixTraits<A>::rows, col = MatrixTraits<A>::columns;
      using MatrixType = Eigen::Matrix<Scalar, col, dim>;
      using ResultType = Eigen::Matrix<Scalar, dim, dim>;
      Eigen::HouseholderQR<MatrixType> QR(adjoint(std::forward<A>(a)));
      ResultType ret;
      if constexpr(col < dim)
      {
        ret << adjoint(QR.matrixQR().template topRows<col>()), Eigen::Matrix<Scalar, dim, dim - col>::Zero();
      }
      else
      {
        ret = adjoint(QR.matrixQR().template topRows<dim>());
      }
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::lower>;
      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_native A>
#else
  template<typename A, std::enable_if_t<eigen_native<A>, int> = 0>
#endif
  constexpr auto
  QR_decomposition(A&& a)
  {
    if constexpr (diagonal_matrix<A> or upper_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto dim = MatrixTraits<A>::rows, col = MatrixTraits<A>::columns;
      using MatrixType = Eigen::Matrix<Scalar, dim, col>;
      using ResultType = Eigen::Matrix<Scalar, col, col>;
      Eigen::HouseholderQR<MatrixType> QR(std::forward<A>(a));
      ResultType ret;
      if constexpr(dim < col)
      {
        ret << QR.matrixQR().template topRows<dim>(), Eigen::Matrix<Scalar, col - dim, col>::Zero();
      }
      else
      {
        ret = QR.matrixQR().template topRows<col>();
      }
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::upper>;
      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects vertically.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((MatrixTraits<V>::columns == MatrixTraits<Vs>::columns) and ...)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
      constexpr auto cols = MatrixTraits<V>::columns;
      Eigen::Matrix<typename MatrixTraits<V>::Scalar, rows, cols> m;
      ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
      return m;
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects horizontally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    ((MatrixTraits<V>::rows == MatrixTraits<Vs>::rows) and ...)
#ifndef __cpp_concepts
      , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto rows = MatrixTraits<V>::rows;
      constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
      Eigen::Matrix<typename MatrixTraits<V>::Scalar, rows, cols> m;
      ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
      return m;
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace detail
  {
    // Concatenate one or more Eigen::MatrixBase objects diagonally.
    template<typename M, typename ... Vs, std::size_t ... ints>
    void
    concatenate_diagonal_impl(M& m, const std::tuple<Vs...>& vs, std::index_sequence<ints...>)
    {
      ((m << std::get<0>(vs)), ..., [&]
      {
        constexpr auto row = (ints + 1) / sizeof...(Vs);
        constexpr auto col = (ints + 1) % sizeof...(Vs);
        constexpr auto row_size = MatrixTraits<decltype(std::get<row>(vs))>::rows;
        constexpr auto col_size = MatrixTraits<decltype(std::get<col>(vs))>::columns;
        if constexpr (row == col) return std::get<row>(vs);
        else return Eigen::Matrix<typename M::Scalar, row_size, col_size>::Zero();
      }());
    }
  }


  /**
   * \brief Concatenate one or more Eigen::MatrixBase objects diagonally.
   * \todo Add special cases for all ZeroMatrix and all ConstantMatrix
   */
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires
#else
  template<typename V, typename ... Vs, std::enable_if_t<
#endif
    ((eigen_matrix<V> or eigen_self_adjoint_expr<V> or eigen_triangular_expr<V> or
      eigen_diagonal_expr<V> or from_euclidean_expr<V>) and ... and
    (eigen_matrix<Vs> or eigen_self_adjoint_expr<Vs> or eigen_triangular_expr<Vs> or
      eigen_diagonal_expr<Vs> or from_euclidean_expr<Vs>)) and
    (not (eigen_diagonal_expr<V> and ... and eigen_diagonal_expr<Vs>)) and
    (not (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)) and
    (not (eigen_self_adjoint_expr<V> and ... and eigen_self_adjoint_expr<Vs>)) and
    (not (eigen_triangular_expr<V> and ... and eigen_triangular_expr<Vs>))
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
      constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
      Eigen::Matrix<typename MatrixTraits<V>::Scalar, rows, cols> m;
      detail::concatenate_diagonal_impl(m, std::forward_as_tuple(v, vs...),
        std::make_index_sequence<(sizeof...(vs) + 1) * (sizeof...(vs) + 1) - 1> {});
      return m;
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace internal
  {
    /// Make a tuple containing an Eigen matrix (general case).
    template<typename F, typename RC, typename CC, typename Arg>
    auto
    make_split_tuple(Arg&& arg)
    {
      auto val = F::template call<RC, CC>(std::forward<Arg>(arg));
      return std::tuple<const decltype(val)> {std::move(val)};
    }


    /// Make a tuple containing an Eigen::Block.
    template<typename F, typename RC, typename CC, typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto
    make_split_tuple(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& arg)
    {
      using NonConstBlock = Eigen::Block<std::remove_const_t<XprType>, BlockRows, BlockCols, InnerPanel>;
      // A const_cast is necessary, because a const Eigen::Block cannot be inserted into a tuple.
      auto& xpr = const_cast<std::remove_const_t<XprType>&>(arg.nestedExpression());
      auto val = F::template call<RC, CC>(NonConstBlock(xpr, arg.startRow(), arg.startCol()));
      return std::tuple<const decltype(val)> {std::move(val)};
    }

  }


  /// Split a matrix vertically.
  /// \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// \tparam euclidean Whether coefficients RC and RCs are transformed to Euclidean space.
  /// \tparam RC Coefficients for the first cut.
  /// \tparam RCs Coefficients for each of the second and subsequent cuts.
#ifdef __cpp_concepts
  template<typename F, bool euclidean, typename RC, typename...RCs, eigen_matrix Arg>
#else
  template<typename F, bool euclidean, typename RC, typename...RCs,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    constexpr auto RC_size = euclidean ? RC::euclidean_dimensions : RC::dimensions;
    static_assert((RC_size + ... + (euclidean ? RCs::euclidean_dimensions : RCs::dimensions)) <= MatrixTraits<Arg>::rows);
    using CC = Axes<MatrixTraits<Arg>::columns>;
    constexpr Eigen::Index dim1 = RC_size, dim2 = std::decay_t<Arg>::RowsAtCompileTime - dim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };
    // \todo Can g be replaced by make_self_contained?
    if constexpr (sizeof...(RCs) > 0)
    {
      auto split1 = g(arg.template topRows<dim1>());
      auto split2 = g(arg.template bottomRows<dim2>());
      return std::tuple_cat(
        internal::make_split_tuple<F, RC, CC>(std::move(split1)),
        split_vertical<F, euclidean, RCs...>(std::move(split2)));
    }
    else if constexpr (dim1 < std::decay_t<Arg>::RowsAtCompileTime)
    {
      return internal::make_split_tuple<F, RC, CC>(g(arg.template topRows<dim1>()));
    }
    else
    {
      return internal::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }

  /// Split a matrix vertically (case in which there is no split).
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix vertically.
  /// \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// \tparam RCs Coefficients for each of the cuts.
#ifdef __cpp_concepts
  template<typename F, coefficients RC, coefficients...RCs, eigen_matrix Arg> requires (not coefficients<F>)
#else
  template<typename F, typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    not coefficients<F> and (coefficients<RC> and ... and coefficients<RCs>), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, false, RC, RCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix vertically.
  /// \tparam RC Coefficients for the first cut.
  /// \tparam RCs Coefficients for the second and subsequent cuts.
#ifdef __cpp_concepts
  template<coefficients RC, coefficients...RCs, eigen_matrix Arg>
#else
  template<typename RC, typename...RCs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (coefficients<RC> and ... and coefficients<RCs>), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, RC, RCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix vertically.
  /// \tparam cut Number of rows in the first cut.
  /// \tparam cuts Numbers of rows in the second and subsequent cuts.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::rows);
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a matrix horizontally and invoke function F on each segment, returning a tuple.
  /// \tparam CC Coefficients for the first cut.
  /// \tparam CCs Coefficients for each of the second and subsequent cuts.
  /// \tparam F An object having a static call() method to which the result is applied before creating the tuple.
#ifdef __cpp_concepts
  template<typename F, coefficients CC, coefficients...CCs, eigen_matrix Arg> requires (not coefficients<F>)
#else
  template<typename F, typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((CC::dimensions + ... + CCs::dimensions) <= MatrixTraits<Arg>::columns);
    using RC = Axes<MatrixTraits<Arg>::rows>;
    constexpr Eigen::Index dim1 = CC::dimensions, dim2 = std::decay_t<Arg>::ColsAtCompileTime - dim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };
    if constexpr(sizeof...(CCs) > 0)
    {
      auto split1 = g(arg.template leftCols<dim1>());
      auto split2 = g(arg.template rightCols<dim2>());
      return std::tuple_cat(
        internal::make_split_tuple<F, RC, CC>(std::move(split1)),
        split_horizontal<F, CCs...>(std::move(split2)));
    }
    else if constexpr(dim1 < std::decay_t<Arg>::ColsAtCompileTime)
    {
      return internal::make_split_tuple<F, RC, CC>(g(arg.template leftCols<dim1>()));
    }
    else
    {
      return internal::make_split_tuple<F, RC, CC>(std::forward<Arg>(arg));
    }
  }

  /// Split a matrix horizontally (case in which there is no split).
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, eigen_matrix Arg>
#else
  template<typename F = OpenKalman::internal::default_split_function, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix horizontally.
  /// \tparam CC Coefficients for the first cut.
  /// \tparam CCs Coefficients for the second and subsequent cuts.
#ifdef __cpp_concepts
  template<coefficients CC, coefficients...CCs, eigen_matrix Arg>
#else
  template<typename CC, typename...CCs, typename Arg,
    std::enable_if_t<eigen_matrix<Arg> and coefficients<CC>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, CC, CCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix horizontally.
  /// \tparam cut Number of columns in the first cut.
  /// \tparam cuts Numbers of columns in the second and subsequent cuts.
  /// \tparam F An object having a static call() method to which the result is applied before creating the tuple.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a matrix diagonally and invoke function F on each segment, returning a tuple. Must be a square matrix.
  /// \tparam F An object having a static call() method to which the result is applied before creating the tuple.
  /// \tparam C Coefficients for the first cut.
  /// \tparam Cs Coefficients for each of the second and subsequent cuts.
#ifdef __cpp_concepts
  template<typename F, bool euclidean, typename C, typename...Cs, eigen_matrix Arg>
#else
  template<typename F, bool euclidean, typename C, typename...Cs,
    typename Arg, std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    constexpr auto RC_size = euclidean ? C::euclidean_dimensions : C::dimensions;
    static_assert((RC_size + ... + (euclidean ? Cs::euclidean_dimensions : Cs::dimensions)) <= MatrixTraits<Arg>::rows);
    static_assert((C::dimensions + ... + Cs::dimensions) <= MatrixTraits<Arg>::columns);
    constexpr Eigen::Index rdim1 = RC_size, rdim2 = std::decay_t<Arg>::RowsAtCompileTime - rdim1;
    constexpr Eigen::Index cdim1 = C::dimensions, cdim2 = std::decay_t<Arg>::RowsAtCompileTime - cdim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return make_native_matrix(std::forward<decltype(m)>(m));
    };
    if constexpr(sizeof...(Cs) > 0)
    {
      auto split1 = g(arg.template topLeftCorner<rdim1, cdim1>());
      auto split2 = g(arg.template bottomRightCorner<rdim2, cdim2>());
      return std::tuple_cat(
        internal::make_split_tuple<F, C, C>(std::move(split1)),
        split_diagonal<F, euclidean, Cs...>(std::move(split2)));
    }
    else if constexpr(rdim1 < std::decay_t<Arg>::RowsAtCompileTime)
    {
      return internal::make_split_tuple<F, C, C>(g(arg.template topLeftCorner<rdim1, cdim1>()));
    }
    else
    {
      return internal::make_split_tuple<F, C, C>(std::forward<Arg>(arg));
    }
  }

  /// Split a matrix vertically (case in which there is no split).
#ifdef __cpp_concepts
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false, eigen_matrix Arg>
    requires (not coefficients<F>)
#else
  template<typename F = OpenKalman::internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<eigen_matrix<Arg> and not coefficients<F>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix vertically.
  /// \tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// \tparam C Coefficients for the first cut.
  /// \tparam Cs Coefficients for each of the second and subsequent cuts.
#ifdef __cpp_concepts
  template<typename F, coefficients C, coefficients...Cs, eigen_matrix Arg> requires (not coefficients<F>)
#else
  template<typename F, typename C, typename...Cs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    not coefficients<F> and (coefficients<C> and ... and coefficients<Cs>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, false, C, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix diagonally. Must be a square matrix.
  /// \tparam C Coefficients for the first cut.
  /// \tparam Cs Coefficients for the second and subsequent cuts.
#ifdef __cpp_concepts
  template<coefficients C, coefficients...Cs, eigen_matrix Arg>
#else
  template<typename C, typename...Cs, typename Arg, std::enable_if_t<eigen_matrix<Arg> and
    (coefficients<C> and ... and coefficients<Cs>), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, C, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix diagonally. Must be a square matrix.
  /// \tparam cut Number of rows and columns in the first cut.
  /// \tparam cuts Numbers of rows and columns in the second and subsequent cuts.
  /// \tparam F An object having a static call() method to which the result is applied before creating the tuple.
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, eigen_matrix Arg>
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg,
    std::enable_if_t<eigen_matrix<Arg>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Get element (i, j) of matrix arg
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  inline auto
  get_element(const Arg& arg, const std::size_t i, const std::size_t j)
  {
    return arg.coeff(i, j);
  }


  /// Get element (i) of one-column matrix arg
#ifdef __cpp_concepts
  template<eigen_native Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  get_element(const Arg& arg, const std::size_t i)
  {
    return arg.coeff(i);
  }


  /// Set element (i, j) of matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_native Arg, typename Scalar> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    (static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit))
#else
  template<typename Arg, typename Scalar,
    std::enable_if_t<eigen_native<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
    static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    arg(i, j) = s;
  }


  /// Set element (i) of one-column matrix arg to s.
#ifdef __cpp_concepts
  template<eigen_native Arg, typename Scalar> requires (not std::is_const_v<std::remove_reference_t<Arg>>) and
    column_vector<Arg> and (static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit))
#else
  template<typename Arg, typename Scalar,
    std::enable_if_t<eigen_native<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      column_vector<Arg> and static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit), int> = 0>
#endif
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    arg(i) = s;
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    constexpr auto r = Eigen::internal::traits<std::decay_t<Arg>>::RowsAtCompileTime;
    return make_self_contained<Arg>(std::forward<Arg>(arg).template block<r, 1>(0, index));
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_native Arg> requires (index < MatrixTraits<Arg>::columns)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
    eigen_native<Arg> and (index < MatrixTraits<Arg>::columns), int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    if constexpr (column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return column(std::forward<Arg>(arg), index);
    }
  }


  namespace detail
  {
    template<bool index, std::size_t i, typename Arg, typename Function>
    inline Arg& do_one_column(Arg& arg, const Function& f)
    {
      auto c = column<i>(arg);
      if constexpr (index) f(c, i); else f(c);
      return arg;
    };


    template<bool index, typename Arg, typename Function, std::size_t ... ints>
    inline Arg& do_columnwise_impl(Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      return (do_one_column<index, ints>(arg, f), ...);
    };


    template<typename Arg, typename Function, std::size_t ... ints>
    inline auto cat_columnwise_impl(const Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      using ResultType = decltype(f(column<0>(arg)));
      if constexpr (euclidean_expr<ResultType>)
      {
        auto res = concatenate_horizontal(nested_matrix(f(column<ints>(arg)))...);
        return MatrixTraits<ResultType>::make(std::move(res));
      }
      else
      {
        return concatenate_horizontal(f(column<ints>(arg))...);
      }
    };


    template<typename Arg, typename Function, std::size_t ... ints>
    inline auto cat_columnwise_index_impl(const Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      using ResultType = decltype(f(column<0>(arg), 0));
      if constexpr (euclidean_expr<ResultType>)
      {
        auto res = concatenate_horizontal(nested_matrix(f(column<ints>(arg), ints))...);
        return MatrixTraits<ResultType>::make(std::move(res));
      }
      else
      {
        return concatenate_horizontal(f(column<ints>(arg), ints)...);
      }
    };


    template<std::size_t i, typename Function>
    constexpr auto cat_columnwise_dummy_function(const Function& f) { return f(); };


    template<typename Function, std::size_t ... ints>
    inline auto cat_columnwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_horizontal(cat_columnwise_dummy_function<ints>(f)...);
    };


    template<typename Function, std::size_t ... ints>
    inline auto cat_columnwise_index_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_horizontal(f(ints)...);
    };

  }

#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_invocable_v<const Function&, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& >, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    return detail::do_columnwise_impl<false>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_invocable_v<Function, std::decay_t<decltype(column(std::declval<Arg>(), 0))>&, std::size_t>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    return detail::do_columnwise_impl<true>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Arg& arg, const Function& f) { {f(column(arg, 0))} -> column_vector; }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    return detail::cat_columnwise_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Arg& arg, const Function& f, std::size_t i) { {f(column(arg, 0), i)} -> column_vector; }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    return detail::cat_columnwise_index_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> eigen_matrix; {f()} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<eigen_matrix<std::invoke_result_t<Function>> and
    column_vector<std::invoke_result_t<Function>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    return detail::cat_columnwise_impl(f, std::make_index_sequence<count>());
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> eigen_matrix; {f(i)} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    eigen_matrix<std::invoke_result_t<Function, std::size_t>> and
    column_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    return detail::cat_columnwise_index_impl(f, std::make_index_sequence<count>());
  }


  ////

#ifdef __cpp_concepts
  template<eigen_native Arg, typename Function> requires element_settable<Arg, 2> and
    (requires(Function f, typename MatrixTraits<Arg>::Scalar& s) { {f(s)} -> std::same_as<void>; } or
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s) {
      {f(s)} -> std::same_as<typename MatrixTraits<Arg>::Scalar&>;
    }) and
    (not requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t& i, std::size_t& j) { f(s, i, j); })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_native<Arg> and element_settable<Arg, 2> and
    (std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>, void> or
    std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>,
      typename MatrixTraits<Arg>::Scalar&>), int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    for (std::size_t j = 0; j < column_count(arg); j++)
    {
      for (std::size_t i = 0; i < row_count(arg); i++)
      {
        f(arg(i, j));
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_native Arg, typename Function> requires element_settable<Arg, 2> and
    (requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::same_as<void>;
    } or
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::same_as<typename MatrixTraits<Arg>::Scalar&>;
    })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_native<Arg> and element_settable<Arg, 2> and
    (std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      void> or
    std::is_same_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      typename MatrixTraits<Arg>::Scalar&>), int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(Arg& arg, Function&& f)
  {
    for (std::size_t j = 0; j < column_count(arg); j++)
    {
      for (std::size_t i = 0; i < row_count(arg); i++)
      {
        f(arg(i, j), i, j);
      }
    }
    return arg;
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s) {
      {f(s)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    } and
    (not requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    })
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename MatrixTraits<Arg>::Scalar&>,
      const typename MatrixTraits<Arg>::Scalar> and
    (not std::is_invocable_v<Function, typename MatrixTraits<Arg>::Scalar&, std::size_t&, std::size_t&>), int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, Function&& f)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    return make_self_contained<Arg>(arg.unaryExpr([fun = std::forward<Function>(f)] (const Scalar& x) {
      if constexpr (std::is_invocable_v<Function, const Scalar&>) return fun(x);
      else { Scalar xx = x; return fun(xx); }
    }));
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires element_gettable<Arg, 2> and
    requires(Function f, typename MatrixTraits<Arg>::Scalar& s, std::size_t i, std::size_t j) {
      {f(s, i, j)} -> std::convertible_to<const typename MatrixTraits<Arg>::Scalar&>;
    }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and element_gettable<Arg, 2> and
    std::is_convertible_v<std::invoke_result_t<Function,
      typename MatrixTraits<Arg>::Scalar&, std::size_t, std::size_t>,
      const typename MatrixTraits<Arg>::Scalar&>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Arg&& arg, Function&& f)
  {
    return native_matrix_t<Arg>::NullaryExpr(
      [m = std::forward<Arg>(arg), fun = std::forward<Function>(f)] (Eigen::Index i, Eigen::Index j) {
        std::size_t ix = i, jx = j;
        return fun(get_element(m, ix, jx), ix, jx);
      });
  }


#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns, typename Function> requires
    requires { typename Eigen::NumTraits<std::invoke_result_t<Function>>; }
#else
  template<std::size_t rows, std::size_t columns, typename Function,
    std::enable_if_t<std::is_void_v<std::void_t<Eigen::NumTraits<std::invoke_result_t<Function>>>>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    using Scalar = std::invoke_result_t<Function>;
    using Mat = eigen_matrix_t<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)>;
    if constexpr (std::is_lvalue_reference_v<Function>)
      return native_matrix_t<Mat>::NullaryExpr(std::cref(f));
    else
      return native_matrix_t<Mat>::NullaryExpr([f = std::move(f)] () { return f(); });
  }


#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns, typename Function> requires
    requires { typename Eigen::NumTraits<std::invoke_result_t<Function, std::size_t&, std::size_t&>>; }
#else
  template<std::size_t rows, std::size_t columns, typename Function,
    typename = std::void_t<Eigen::NumTraits<std::invoke_result_t<Function, std::size_t&, std::size_t&>>>>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    using Scalar = std::invoke_result_t<Function, std::size_t, std::size_t>;
    using Mat = eigen_matrix_t<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)>;
    if constexpr (std::is_lvalue_reference_v<Function>)
      return native_matrix_t<Mat>::NullaryExpr(std::cref(f));
    else
      return native_matrix_t<Mat>::NullaryExpr([f = std::move(f)] (std::size_t i, std::size_t j) { return f(i, j); });
  }


  namespace detail
  {
    template<typename Scalar, template<typename> typename distribution_type, typename random_number_engine>
    inline auto
    get_rnd(const typename distribution_type<Scalar>::param_type& params)
    {
      static std::random_device rd;
      static random_number_engine rng {rd()};
      static distribution_type<Scalar> dist;
      return dist(rng, params);
    }


#ifdef __cpp_concepts
    template<typename Scalar, template<typename> typename, typename> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, template<typename> typename, typename,
      std::enable_if_t<std::is_arithmetic_v<Scalar>, int> = 0>
#endif
    constexpr auto
    get_rnd(Scalar s) { return s; }

  } // namespace detail


  /**
   * \brief Fill an Eigen matrix with random values selected from a random distribution.
   * \details The Gaussian distribution has mean zero and a scalar standard deviation sigma (== 1, if not specified).
   **/
#ifdef __cpp_concepts
  template<
    eigen_matrix ReturnType,
    template<typename> typename distribution_type = std::normal_distribution,
    std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Params>
#else
  template<
    typename ReturnType,
    template<typename> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<eigen_matrix<ReturnType>, int> = 0>
#endif
  inline auto
  randomize(Params&& ... params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    constexpr auto rows = MatrixTraits<ReturnType>::rows;
    constexpr auto cols = MatrixTraits<ReturnType>::columns;
    using Ps = typename distribution_type<Scalar>::param_type;
    if constexpr (std::is_constructible_v<Ps, Params...>)
    {
      return make_self_contained(ReturnType::NullaryExpr([ps = Ps {params...}] {
        return detail::get_rnd<Scalar, distribution_type, random_number_engine>(ps);
      }));
    }
    else if constexpr (sizeof...(Params) == rows * cols)
    {
      return MatrixTraits<ReturnType>::make(
        detail::get_rnd<Scalar, distribution_type, random_number_engine>(std::forward<Params>(params))...);
    }
    else
    {
      static_assert(sizeof...(Params) == rows,
        "Params... must be (1) a parameter set or list of parameter sets, "
        "(2) a list of parameter sets, one for each row, or (3) a list of parameter sets, one for each coefficient.");
      return apply_columnwise<cols>([&] {
        using ReturnTypeCol = native_matrix_t<ReturnType, rows, 1>;
        return MatrixTraits<ReturnTypeCol>::make(
          detail::get_rnd<Scalar, distribution_type, random_number_engine>(std::forward<Params>(params))...);
      });
    }

  }


}

#endif //OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP
