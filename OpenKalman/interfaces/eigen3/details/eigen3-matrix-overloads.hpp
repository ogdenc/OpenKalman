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
  template<arithmetic_or_complex Scalar, std::size_t rows, std::size_t columns = 1, std::convertible_to<Scalar> ... Args>
  requires ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or (rows == 0 and sizeof...(Args) % columns == 0))
#else
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename Scalar, std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    arithmetic_or_complex<Scalar> and (std::is_convertible_v<Args, Scalar> and ...) and
    ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
      (columns == 0 and sizeof...(Args) % rows == 0) or (rows == 0 and sizeof...(Args) % columns == 0)), int> = 0>
// See below for pragma pop...
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using M = Eigen3::eigen_matrix_t<Scalar, rows, columns>;
    return MatrixTraits<M>::make(static_cast<const Scalar>(args)...);
  }


  /**
   * \overload
   * \brief Make a native Eigen matrix from a list of coefficients in row-major order.
   */
#ifdef __cpp_concepts
  template<std::size_t rows, std::size_t columns = 1, arithmetic_or_complex ... Args>
  requires ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
    (columns == 0 and sizeof...(Args) % rows == 0) or (rows == 0 and sizeof...(Args) % columns == 0))
#else
  template<std::size_t rows, std::size_t columns = 1, typename ... Args, std::enable_if_t<
    (arithmetic_or_complex<Args> and ...) and ((rows != 0 and columns != 0 and sizeof...(Args) == rows * columns) or
        (columns == 0 and sizeof...(Args) % rows == 0) or (rows == 0 and sizeof...(Args) % columns == 0)), int> = 0>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Scalar = std::common_type_t<Args...>;
    return make_native_matrix<Scalar, rows, columns>(args...);
  }


  /// Make a native Eigen 1-column vector from a list of coefficients in row-major order.
#ifdef __cpp_concepts
  template<arithmetic_or_complex ... Args>
#else
#pragma GCC diagnostic pop
  template<typename ... Args, std::enable_if_t<(arithmetic_or_complex<Args> and ...), int> = 0>
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
  constexpr std::size_t
  row_count(Arg&& arg)
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
  constexpr std::size_t
  column_count(Arg&& arg)
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
  template<eigen_matrix Arg> requires one_by_one_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_matrix<Arg> and one_by_one_matrix<Arg>, int> = 0>
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
    using Scalar = typename MatrixTraits<Arg>::Scalar;

    if constexpr (identity_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        std::size_t dim = row_count(arg);
        assert(dim == column_count(arg));
        if constexpr (dynamic_columns<Arg>)
          return ConstantMatrix<Scalar, 1, 0, 1> {dim};
        else
          return ConstantMatrix<Scalar, 1, MatrixTraits<Arg>::columns, 1> {};
      }
      else
      {
        return ConstantMatrix<Scalar, 1, MatrixTraits<Arg>::rows, 1> {};
      }
    }
    else if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
      {
        std::size_t dim = row_count(arg);
        assert(dim == column_count(arg));
        if constexpr (dynamic_columns<Arg>)
          return ZeroMatrix<Scalar, 0, 1>(dim);
        else
          return ZeroMatrix<Scalar, MatrixTraits<Arg>::columns, 1>(dim, 1);
      }
      else
      {
        return ZeroMatrix<Scalar, MatrixTraits<Arg>::rows, 1> {};
      }
    }
    else
    {
      using Ret = decltype(std::forward<Arg>(arg).diagonal());

      if constexpr (dynamic_shape<Ret> and (not dynamic_rows<Arg> or not dynamic_columns<Arg>))
      {
        constexpr std::size_t dim = dynamic_rows<Arg> ? MatrixTraits<Arg>::columns : MatrixTraits<Arg>::rows;
        return eigen_matrix_t<Scalar, dim, 1> {std::forward<Arg>(arg).diagonal()};
      }
      else
      {
        return std::forward<Arg>(arg).diagonal();
      }
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
    if constexpr (self_adjoint_matrix<Arg> and not complex_number<typename MatrixTraits<Arg>::Scalar>)
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
      if constexpr (self_adjoint_matrix<Arg>)
      {
        return std::forward<Arg>(arg);
      }
      else
      {
        return std::forward<Arg>(arg).transpose();
      }
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  inline auto
  determinant(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
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
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return arg.determinant();
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
    {
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
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
      if constexpr (dynamic_shape<Arg>) assert(row_count(arg) == column_count(arg));
      return arg.trace();
    }
  }


  /// Solve the equation AX = B for X. A is an invertible square matrix. (Does not check that A is invertible.)
  /// Uses the square LU decomposition.
#ifdef __cpp_concepts
  template<eigen_native A, eigen_matrix B> requires (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or MatrixTraits<A>::rows == MatrixTraits<B>::rows)
#else
  template<typename A, typename B, std::enable_if_t<eigen_native<A> and eigen_matrix<B> and
    (dynamic_shape<A> or square_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<B> or MatrixTraits<A>::rows == MatrixTraits<B>::rows), int> = 0>
#endif
  constexpr auto
  solve(const A& a, const B& b)
  {
    using M = eigen_matrix_t<typename MatrixTraits<B>::Scalar, MatrixTraits<A>::rows, MatrixTraits<B>::columns>;
    if constexpr (one_by_one_matrix<A>)
    {
      if constexpr (dynamic_shape<B>) assert(row_count(b) == 1 and column_count(b) == 1);
      return M {trace(b)/trace(a)};
    }
    else
    {
      if constexpr (dynamic_shape<A>) assert(row_count(a) == column_count(a));
      if constexpr (dynamic_rows<A> or dynamic_rows<B>) assert(row_count(a) == row_count(b));
      return M {a.lu().solve(b)};
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
      return arg.rowwise().sum() / column_count(arg);
    }
  }


  /// Create a row vector by taking the mean of each column in a set of row vectors.
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_rows(Arg&& arg) noexcept
  {
    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return arg.colwise().sum() / row_count(arg);
    }
  }


  namespace detail
  {
    template<typename A>
    constexpr auto
    QR_decomp_impl(A&& a)
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto rows = MatrixTraits<A>::rows;
      constexpr auto cols = MatrixTraits<A>::columns;
      using MatrixType = Eigen3::eigen_matrix_t<Scalar, rows, cols>;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;

      Eigen::HouseholderQR<MatrixType> QR {std::forward<A>(a)};

      if constexpr (dynamic_columns<A>)
      {
        auto rt_cols = column_count(a);

        ResultType ret {rt_cols, rt_cols};

        if constexpr (dynamic_rows<A>)
        {
          auto rt_rows = row_count(a);

          if (rt_rows < rt_cols)
            ret << QR.matrixQR().topRows(rt_rows),
              Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(rt_cols - rt_rows, rt_cols);
          else
            ret = QR.matrixQR().topRows(rt_cols);
        }
        else
        {
          if (rows < rt_cols)
            ret << QR.matrixQR().template topRows<rows>(),
              Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(rt_cols - rows, rt_cols);
          else
            ret = QR.matrixQR().topRows(rt_cols);
        }

        return ret;
      }
      else
      {
        ResultType ret;

        if constexpr (dynamic_rows<A>)
        {
          auto rt_rows = row_count(a);

          if (rt_rows < cols)
            ret << QR.matrixQR().topRows(rt_rows), Eigen3::eigen_matrix_t<Scalar, 0, 0>::Zero(cols - rt_rows, cols);
          else
            ret = QR.matrixQR().template topRows<cols>();
        }
        else
        {
          if constexpr (rows < cols)
            ret << QR.matrixQR().template topRows<rows>(), Eigen3::eigen_matrix_t<Scalar, cols - rows, cols>::Zero();
          else
            ret = QR.matrixQR().template topRows<cols>();
        }

        return ret;
      }
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
    if constexpr (lower_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto rows = MatrixTraits<A>::rows;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, rows, rows>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::lower>;

      ResultType ret = adjoint(detail::QR_decomp_impl(adjoint(std::forward<A>(a))));

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
    if constexpr (upper_triangular_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto cols = MatrixTraits<A>::columns;
      using ResultType = Eigen3::eigen_matrix_t<Scalar, cols, cols>;
      using TType = typename MatrixTraits<ResultType>::template TriangularMatrixFrom<TriangleType::upper>;

      ResultType ret = detail::QR_decomp_impl(std::forward<A>(a));

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
    ((dynamic_columns<V> or dynamic_columns<Vs> or MatrixTraits<V>::columns == MatrixTraits<Vs>::columns) and ...)
#ifndef __cpp_concepts
    , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = typename MatrixTraits<V>::Scalar;

      if constexpr ((dynamic_columns<V> and ... and dynamic_columns<Vs>))
      {
        auto cols = column_count(v);
        assert(((cols == column_count(vs)) and ...));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          Eigen3::eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto cols = std::max({MatrixTraits<V>::columns, MatrixTraits<Vs>::columns...});

        static_assert(((dynamic_columns<V> or MatrixTraits<V>::columns == cols) and ... and
          (dynamic_columns<Vs> or MatrixTraits<Vs>::columns == cols)));

        if constexpr ((dynamic_rows<V> or ... or dynamic_rows<Vs>))
        {
          auto rows = (row_count(v) + ... + row_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto rows = (MatrixTraits<V>::rows + ... + MatrixTraits<Vs>::rows);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
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
    ((dynamic_rows<V> or dynamic_rows<Vs> or MatrixTraits<V>::rows == MatrixTraits<Vs>::rows) and ...)
#ifndef __cpp_concepts
      , int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using Scalar = typename MatrixTraits<V>::Scalar;

      if constexpr ((dynamic_rows<V> and ... and dynamic_rows<Vs>))
      {
        auto rows = row_count(v);
        assert(((rows == row_count(vs)) and ...));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (column_count(v) + ... + column_count(vs));
          Eigen3::eigen_matrix_t<Scalar, 0, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          Eigen3::eigen_matrix_t<Scalar, 0, cols> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
      else
      {
        constexpr auto rows = std::max({MatrixTraits<V>::rows, MatrixTraits<Vs>::rows...});

        static_assert(((dynamic_rows<V> or MatrixTraits<V>::rows == rows) and ... and
          (dynamic_rows<Vs> or MatrixTraits<Vs>::rows == rows)));

        if constexpr ((dynamic_columns<V> or ... or dynamic_columns<Vs>))
        {
          auto cols = (column_count(v) + ... + column_count(vs));
          Eigen3::eigen_matrix_t<Scalar, rows, 0> m {rows, cols};
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
        else
        {
          constexpr auto cols = (MatrixTraits<V>::columns + ... + MatrixTraits<Vs>::columns);
          Eigen3::eigen_matrix_t<Scalar, rows, cols> m;
          ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
          return m;
        }
      }
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
        else return Eigen3::eigen_matrix_t<typename M::Scalar, row_size, col_size>::Zero();
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
      Eigen3::eigen_matrix_t<typename MatrixTraits<V>::Scalar, rows, cols> m;
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


  /// Return row <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_native Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg>, int> = 0>
#endif
  inline auto
  row(Arg&& arg, const std::size_t index)
  {
    constexpr auto c = Eigen::internal::traits<std::decay_t<Arg>>::ColsAtCompileTime;
    return make_self_contained<Arg>(std::forward<Arg>(arg).template block<1, c>(index, 0));
  }


  /// Return row <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<std::size_t index, eigen_native Arg> requires (index < MatrixTraits<Arg>::rows)
#else
  template<std::size_t index, typename Arg, std::enable_if_t<
      eigen_native<Arg> and (index < MatrixTraits<Arg>::rows), int> = 0>
#endif
  inline decltype(auto)
  row(Arg&& arg)
  {
    if constexpr (row_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return row(std::forward<Arg>(arg), index);
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


  namespace detail
  {
    template<bool index, std::size_t i, typename Arg, typename Function>
    inline Arg& do_one_row(Arg& arg, const Function& f)
    {
      auto r = row<i>(arg);
      if constexpr (index) f(r, i); else f(r);
      return arg;
    };


    template<bool index, typename Arg, typename Function, std::size_t ... ints>
    inline Arg& do_rowwise_impl(Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      return (do_one_row<index, ints>(arg, f), ...);
    };


    template<typename Arg, typename Function, std::size_t ... ints>
    inline auto cat_rowwise_impl(const Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(row<ints>(arg))...);
    };


    template<typename Arg, typename Function, std::size_t ... ints>
    inline auto cat_rowwise_index_impl(const Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(row<ints>(arg), ints)...);
    };


    template<std::size_t i, typename Function>
    constexpr auto cat_rowwise_dummy_function(const Function& f) { return f(); };


    template<typename Function, std::size_t ... ints>
    inline auto cat_rowwise_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(cat_rowwise_dummy_function<ints>(f)...);
    };


    template<typename Function, std::size_t ... ints>
    inline auto cat_rowwise_index_impl(const Function& f, std::index_sequence<ints...>)
    {
      return concatenate_vertical(f(ints)...);
    };

  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Function& f, std::decay_t<decltype(row(std::declval<Arg>(), 0))>& row) { f(row); }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_invocable_v<const Function&, std::decay_t<decltype(row(std::declval<Arg>(), 0))>& >, int> = 0>
#endif
  inline Arg&
  apply_rowwise(Arg& arg, const Function& f)
  {
    return detail::do_rowwise_impl<false>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Function& f, std::decay_t<decltype(row(std::declval<Arg>(), 0))>& row, std::size_t i) { f(row, i); }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
    std::is_invocable_v<Function, std::decay_t<decltype(row(std::declval<Arg>(), 0))>&, std::size_t>, int> = 0>
#endif
  inline Arg&
  apply_rowwise(Arg& arg, const Function& f)
  {
    return detail::do_rowwise_impl<true>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Arg& arg, const Function& f) { {f(row(arg, 0))} -> row_vector; }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
      not std::is_void_v<std::invoke_result_t<Function,
        std::decay_t<decltype(row(std::declval<Arg>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Arg& arg, const Function& f)
  {
    return detail::cat_rowwise_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<eigen_matrix Arg, typename Function> requires
    requires(const Arg& arg, const Function& f, std::size_t i) { {f(row(arg, 0), i)} -> row_vector; }
#else
  template<typename Arg, typename Function, std::enable_if_t<eigen_matrix<Arg> and
      not std::is_void_v<std::invoke_result_t<Function,
        std::decay_t<decltype(row(std::declval<Arg>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Arg& arg, const Function& f)
  {
    return detail::cat_rowwise_index_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::rows>());
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> eigen_matrix; {f()} -> row_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<eigen_matrix<std::invoke_result_t<Function>> and
    row_vector<std::invoke_result_t<Function>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f)
  {
    return detail::cat_rowwise_impl(f, std::make_index_sequence<count>());
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> eigen_matrix; {f(i)} -> row_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    eigen_matrix<std::invoke_result_t<Function, std::size_t>> and
    row_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f)
  {
    return detail::cat_rowwise_index_impl(f, std::make_index_sequence<count>());
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
    template<typename random_number_engine>
    struct Rnd
    {
      template<typename distribution_type>
      static inline auto
      get(distribution_type& dist)
      {
        if constexpr (std::is_arithmetic_v<distribution_type>)
        {
          return dist;
        }
        else
        {
          static std::random_device rd;
          static random_number_engine rng {rd()};

#ifdef __cpp_concepts
          static_assert(requires { typename distribution_type::result_type; typename distribution_type::param_type; });
#endif
          return dist(rng);
        }
      }

    };


#ifdef __cpp_concepts
  template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct dist_result_type {};

#ifdef __cpp_concepts
    template<typename T> requires requires { typename T::result_type; typename T::param_type; }
    struct dist_result_type<T> { using type = typename T::result_type; };
#else
    template<typename T>
    struct dist_result_type<T, std::enable_if_t<not std::is_arithmetic_v<T>>> { using type = typename T::result_type; };
#endif

#ifdef __cpp_concepts
  template<typename T> requires std::is_arithmetic_v<T>
  struct dist_result_type<T> { using type = T; };
#else
  template<typename T>
    struct dist_result_type<T, std::enable_if_t<std::is_arithmetic_v<T>>> { using type = T; };
#endif



  } // namespace detail


  /**
   * \brief Fill a fixed-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs a 2-by-2 matrix (m) in which each element is a random value selected
   * based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(N {1.0, 0.3}));
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \tparam Dist A distribution (e.g., std::normal_distribution<double>).
   **/
#ifdef __cpp_concepts
  template<eigen_matrix ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename Dist>
  requires
    (not dynamic_shape<ReturnType>) and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist,
    std::enable_if_t<eigen_matrix<ReturnType> and (not dynamic_shape<ReturnType>)and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(Dist&& dist)
  {
    using D = struct { mutable Dist value; };
    return ReturnType::NullaryExpr([d = D {std::forward<Dist>(dist)}] {
      return detail::Rnd<random_number_engine>::get(d.value);
    });
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape Eigen matrix with random values selected from a single random distribution.
   * \details The following example constructs two 2-by-2 matrices (m, n, and p) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     auto m = randomize(Eigen::Matrix<float, 2, Eigen::Dynamic>, 2, 2, std::normal_distribution<float> {1.0, 0.3}));
   *     auto n = randomize(Eigen::Matrix<double, Eigen::Dynamic, 2>, 2, 2, std::normal_distribution<double> {1.0, 0.3}));
   *     auto p = randomize(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>, 2, 2, std::normal_distribution<double> {1.0, 0.3});
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime. Must match rows of ReturnType if they are fixed.
   * \param columns Number of columns, decided at runtime. Must match columns of ReturnType if they are fixed.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<eigen_matrix ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename Dist>
  requires
    dynamic_shape<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
      (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
    eigen_matrix<ReturnType> and dynamic_shape<ReturnType> and
    (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    if constexpr (not dynamic_rows<ReturnType>) assert(rows == MatrixTraits<ReturnType>::rows);
    if constexpr (not dynamic_columns<ReturnType>) assert(columns == MatrixTraits<ReturnType>::columns);
    using D = struct { mutable Dist value; };
    return ReturnType::NullaryExpr(rows, columns, [d = D {std::forward<Dist>(dist)}] {
      return detail::Rnd<random_number_engine>::get(d.value);
    });
  }


  /**
   * \overload
   * \brief Fill a fixed Eigen matrix with random values selected from multiple random distributions.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *
   *  - One distribution for each row. The following code constructs a 3-by-2 (m) or 2-by-2 (p) matrices
   *  in which elements in each row are selected according to the three (m) or two (p) listed distribution
   *  parameters:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 3, 2>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto p = randomize<Eigen::Matrix<double, 2, 2>>(N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of p, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *
   *  - One distribution for each column. The following code constructs 2-by-3 matrix m
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 3>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   *
   * \tparam ReturnType The return type reflecting the size of the matrix to be filled. The actual result will be
   * a fixed type matrix.
   * \tparam random_number_engine The random number engine.
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
   **/
#ifdef __cpp_concepts
  template<
    eigen_matrix ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename...Dists>
  requires
  (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 1) and
    (((requires { typename std::decay_t<Dists>::result_type;  typename std::decay_t<Dists>::param_type; } or
      std::is_arithmetic_v<std::decay_t<Dists>>) and ... ))and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
    (MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
      MatrixTraits<ReturnType>::rows == sizeof...(Dists) or MatrixTraits<ReturnType>::columns == sizeof...(Dists))
#else
  template<
    typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<
      eigen_matrix<ReturnType> and (not dynamic_shape<ReturnType>) and (sizeof...(Dists) > 1) and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
      (MatrixTraits<ReturnType>::rows * MatrixTraits<ReturnType>::columns == sizeof...(Dists) or
        MatrixTraits<ReturnType>::rows == sizeof...(Dists) or
        MatrixTraits<ReturnType>::columns == sizeof...(Dists)), int> = 0>
#endif
  inline auto
  randomize(Dists&& ... dists)
  {
    using Scalar = std::common_type_t<typename detail::dist_result_type<Dists>::type...>;
    constexpr std::size_t s = sizeof...(Dists);
    constexpr std::size_t rows = MatrixTraits<ReturnType>::rows;
    constexpr std::size_t cols = MatrixTraits<ReturnType>::columns;

    // One distribution for each element
    if constexpr (rows * cols == s)
    {
      using M = eigen_matrix_t<Scalar, rows, cols>;
      return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
    }

    // One distribution for each row
    else if constexpr (rows == s)
    {
      using M = eigen_matrix_t<Scalar, rows, 1>;
      return apply_columnwise<cols>([&] {
        return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
      });
    }

    // One distribution for each column
    else
    {
      static_assert(cols == s);
      using M = eigen_matrix_t<Scalar, 1, cols>;
      return apply_rowwise<rows>([&] {
        return MatrixTraits<M>::make(detail::Rnd<random_number_engine>::get(dists)...);
      });
    }

  }


}

#endif //OPENKALMAN_EIGEN3_MATRIX_OVERLOADS_HPP
