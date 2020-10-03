/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENMATRIXOVERLOADS_H
#define OPENKALMAN_EIGENMATRIXOVERLOADS_H

#include <type_traits>
#include <random>

namespace OpenKalman
{
  /// Convert to strict version of the matrix.
  template<typename Arg, std::enable_if_t<
    is_Eigen_matrix_v<Arg> or
    is_EigenDiagonal_v<Arg> or
    is_EigenSelfAdjointMatrix_v<Arg> or
    is_EigenTriangularMatrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict_matrix(Arg&& arg)
  {
    if constexpr(is_strict_matrix_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using S = typename MatrixTraits<Arg>::Scalar;
      constexpr Eigen::Index rows = MatrixTraits<Arg>::dimension;
      constexpr Eigen::Index cols = MatrixTraits<Arg>::columns;
      return static_cast<Eigen::Matrix<S, rows, cols>>(std::forward<Arg>(arg));
    }
  }


  /// Convert to strict version of the matrix.
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg)
  {
    if constexpr(is_strict_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return strict_matrix(std::forward<Arg>(arg));
    }
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  to_Euclidean(Arg&& arg) noexcept
  {
    static_assert(not is_ToEuclideanExpr_v<Arg>);
    if constexpr(Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return ToEuclideanExpr<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  from_Euclidean(Arg&& arg) noexcept
  {
    static_assert(not is_FromEuclideanExpr_v<Arg>);
    if constexpr(Coefficients::axes_only)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return FromEuclideanExpr<Coefficients, std::decay_t<Arg>>(std::forward<Arg>(arg));
    }
  }


  template<typename Coefficients, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  constexpr decltype(auto)
  wrap_angles(Arg&& arg) noexcept
  {
    if constexpr(Coefficients::axes_only or is_identity_v<Arg> or is_zero_v<Arg>)
    {
      /// @TODO: Add functionality to conditionally wrap zero and identity, depending on wrap min and max.
      return std::forward<Arg>(arg);
    }
    else
    {
      return from_Euclidean<Coefficients>(to_Euclidean<Coefficients>(std::forward<Arg>(arg)));
    }
  }


  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::columns == 1);
    if constexpr(is_1by1_v<Arg>)
      return std::forward<Arg>(arg);
    else
      return EigenDiagonal(std::forward<Arg>(arg));
  }


  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    if constexpr(is_1by1_v<Arg>)
      return std::forward<Arg>(arg);
    else
      return std::forward<Arg>(arg).transpose();
  }


  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    if constexpr(is_1by1_v<Arg>)
      return std::forward<Arg>(arg);
    else
      return std::forward<Arg>(arg).adjoint();
  }


  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    if constexpr(is_1by1_v<Arg>)
      return std::forward<Arg>(arg)(0, 0);
    else
      return std::forward<Arg>(arg).determinant();
  }


  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    if constexpr(is_1by1_v<Arg>)
      return std::forward<Arg>(arg)(0, 0);
    else
      return std::forward<Arg>(arg).trace();
  }


  /// Solve the equation AX = B for X. A is an invertible square matrix. (Does not check that A is invertible.)
  /// Uses the square LU decomposition.
  template<
    typename A, typename B,
    std::enable_if_t<is_Eigen_matrix_v<A>, int> = 0,
    std::enable_if_t<is_Eigen_matrix_v<B>, int> = 0>
  inline auto
  solve(const A& a, const B& b)
  {
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<A>::columns);
    static_assert(MatrixTraits<A>::dimension == MatrixTraits<B>::dimension);
    using M = Eigen::Matrix<typename MatrixTraits<B>::Scalar, MatrixTraits<A>::dimension, MatrixTraits<B>::columns>;
    if constexpr(is_1by1_v<A>)
    {
      return M(b(0, 0)/a(0, 0));
    }
    else
    {
      return M(a.lu().solve(b));
    }
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::columns == 1)
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
  template<typename A, std::enable_if_t<is_native_Eigen_type_v<A>, int> = 0>
  constexpr auto
  LQ_decomposition(A&& a)
  {
    if constexpr(is_diagonal_v<A> or is_lower_triangular_v<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto dim = MatrixTraits<A>::dimension, col = MatrixTraits<A>::columns;
      using MatrixType = Eigen::Matrix<Scalar, col, dim>;
      using ResultType = Eigen::Matrix<Scalar, dim, dim>;
      Eigen::HouseholderQR<MatrixType> QR(std::forward<A>(a).adjoint());
      ResultType ret;
      if constexpr(col < dim)
      {
        ret << QR.matrixQR().template topRows<col>().adjoint(), Eigen::Matrix<Scalar, dim, dim - col>::Zero();
      }
      else
      {
        ret = QR.matrixQR().template topRows<dim>().adjoint();
      }
      using TType = typename MatrixTraits<ResultType>::template TriangularBaseType<TriangleType::lower>;
      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
  template<typename A, std::enable_if_t<is_native_Eigen_type_v<A>, int> = 0>
  constexpr auto
  QR_decomposition(A&& a)
  {
    if constexpr(is_diagonal_v<A> or is_upper_triangular_v<A>)
    {
      return std::forward<A>(a);
    }
    else
    {
      using Scalar = typename MatrixTraits<A>::Scalar;
      constexpr auto dim = MatrixTraits<A>::dimension, col = MatrixTraits<A>::columns;
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
      using TType = typename MatrixTraits<ResultType>::template TriangularBaseType<TriangleType::upper>;
      return MatrixTraits<TType>::make(std::move(ret));
    }
  }


  /// Concatenate one or more Eigen::MatrixBase objects vertically.
  template<
    typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<
      std::disjunction<is_Eigen_matrix<V>, is_EigenSelfAdjointMatrix<V>, is_EigenTriangularMatrix<V>,
        is_EigenDiagonal<V>, is_FromEuclideanExpr<V>>,
      std::disjunction<is_Eigen_matrix<Vs>, is_EigenSelfAdjointMatrix<Vs>, is_EigenTriangularMatrix<Vs>,
        is_EigenDiagonal<Vs>, is_FromEuclideanExpr<V>>...> and
      not std::conjunction_v<is_FromEuclideanExpr<V>, is_FromEuclideanExpr<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      constexpr auto rows = (MatrixTraits<V>::dimension + ... + MatrixTraits<Vs>::dimension);
      constexpr auto cols = MatrixTraits<V>::columns;
      static_assert(((cols == MatrixTraits<Vs>::columns) and ...));
      Eigen::Matrix<typename MatrixTraits<V>::Scalar, rows, cols> m;
      ((m << std::forward<V>(v)), ..., std::forward<Vs>(vs));
      return m;
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /// Concatenate one or more Eigen::MatrixBase objects horizontally.
  template<
    typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<
      std::disjunction<is_Eigen_matrix<V>, is_EigenSelfAdjointMatrix<V>, is_EigenTriangularMatrix<V>,
        is_EigenDiagonal<V>, is_FromEuclideanExpr<V>>,
      std::disjunction<is_Eigen_matrix<Vs>, is_EigenSelfAdjointMatrix<Vs>, is_EigenTriangularMatrix<Vs>,
        is_EigenDiagonal<Vs>, is_FromEuclideanExpr<Vs>>...> and
      not std::conjunction_v<is_FromEuclideanExpr<V>, is_FromEuclideanExpr<Vs>...>, int> = 0>
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      constexpr auto rows = MatrixTraits<V>::dimension;
      static_assert(((rows == MatrixTraits<Vs>::dimension) and ...));
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
    /// Concatenate one or more Eigen::MatrixBase objects diagonally.
    template<typename M, typename ... Vs, std::size_t ... ints>
    void
    concatenate_diagonal_impl(M& m, const std::tuple<Vs...>& vs, std::index_sequence<ints...>)
    {
      ((m << std::get<0>(vs)), ..., [&vs]
      {
        constexpr auto row = (ints + 1) / sizeof...(Vs);
        constexpr auto col = (ints + 1) % sizeof...(Vs);
        constexpr auto row_size = MatrixTraits<decltype(std::get<row>(vs))>::dimension;
        constexpr auto col_size = MatrixTraits<decltype(std::get<col>(vs))>::columns;
        if constexpr(row == col) return std::get<row>(vs);
        else return Eigen::Matrix<typename M::Scalar, row_size, col_size>::Zero();
      }());
    }
  }


  /// Concatenate one or more Eigen::MatrixBase objects diagonally.
  template<
    typename V, typename ... Vs,
    std::enable_if_t<std::conjunction_v<
      std::disjunction<is_Eigen_matrix<V>, is_EigenSelfAdjointMatrix<V>, is_EigenTriangularMatrix<V>,
        is_EigenDiagonal<V>, is_FromEuclideanExpr<V>>,
      std::disjunction<is_Eigen_matrix<Vs>, is_EigenSelfAdjointMatrix<Vs>, is_EigenTriangularMatrix<Vs>,
        is_EigenDiagonal<Vs>, is_FromEuclideanExpr<Vs>>...> and
      not std::conjunction_v<is_FromEuclideanExpr<V>, is_FromEuclideanExpr<Vs>...> and
      not (std::conjunction_v<is_EigenSelfAdjointMatrix<V>, is_EigenSelfAdjointMatrix<Vs>...> or
      (std::conjunction_v<is_EigenTriangularMatrix<V>, is_EigenTriangularMatrix<Vs>...> and
        ((is_upper_triangular_v<V> == is_upper_triangular_v<Vs>) and ... and true))), int> = 0>
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      constexpr auto rows = (MatrixTraits<V>::dimension + ... + MatrixTraits<Vs>::dimension);
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
      return std::tuple {F::template call<RC, CC>(std::forward<Arg>(arg))};
    }

    /// Make a tuple containing an Eigen::Block.
    template<typename F, typename RC, typename CC, typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    auto
    make_split_tuple(Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>&& arg)
    {
      using NonConstBlock = Eigen::Block<std::remove_const_t<XprType>, BlockRows, BlockCols, InnerPanel>;
      // A const_cast is necessary, because a const Eigen::Block cannot be inserted into a tuple.
      auto& xpr = const_cast<std::remove_const_t<XprType>&>(arg.nestedExpression());
      const auto block = F::template call<RC, CC>(NonConstBlock(xpr, arg.startRow(), arg.startCol()));
      return std::tuple {std::move(block)};
    }

    /// Default split function.
    struct default_split_function
    {
      template<typename, typename, typename Arg>
      static constexpr Arg&& call(Arg&& arg) { return std::forward<Arg>(arg); }
    };
  }


  /// Split a matrix vertically.
  /// @tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// @tparam euclidean Whether coefficients RC and RCs are transformed to Euclidean space.
  /// @tparam RC Coefficients for the first cut.
  /// @tparam RCs Coefficients for each of the second and subsequent cuts.
  template<typename F, bool euclidean, typename RC, typename...RCs,
    typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    constexpr auto RC_size = euclidean ? RC::dimension : RC::size;
    static_assert((RC_size + ... + (euclidean ? RCs::dimension : RCs::size)) <= MatrixTraits<Arg>::dimension);
    using CC = Axes<MatrixTraits<Arg>::columns>;
    constexpr Eigen::Index dim1 = RC_size, dim2 = std::decay_t<Arg>::RowsAtCompileTime - dim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return strict_matrix(std::forward<decltype(m)>(m));
    };
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
  template<typename F = internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix vertically.
  /// @tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// @tparam RCs Coefficients for each of the cuts.
  template<typename F, typename RC, typename...RCs, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    not is_coefficient_v<F> and std::conjunction_v<is_coefficient<RC>, is_coefficient<RCs>...>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, false, RC, RCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix vertically.
  /// @tparam RC Coefficients for the first cut.
  /// @tparam RCs Coefficients for the second and subsequent cuts.
  template<typename RC, typename...RCs, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::conjunction_v<is_coefficient<RC>, is_coefficient<RCs>...>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<internal::default_split_function, RC, RCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix vertically.
  /// @tparam cut Number of rows in the first cut.
  /// @tparam cuts Numbers of rows in the second and subsequent cuts.
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::dimension);
    return split_vertical<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a matrix horizontally and invoke function F on each segment, returning a tuple.
  /// @tparam CC Coefficients for the first cut.
  /// @tparam CCs Coefficients for each of the second and subsequent cuts.
  /// @tparam F An object having a static call() method to which the result is applied before creating the tuple.
  template<typename F, typename CC, typename...CCs, typename Arg,
    std::enable_if_t<is_Eigen_matrix_v<Arg> and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((CC::size + ... + CCs::size) <= MatrixTraits<Arg>::columns);
    using RC = Axes<MatrixTraits<Arg>::dimension>;
    constexpr Eigen::Index dim1 = CC::size, dim2 = std::decay_t<Arg>::ColsAtCompileTime - dim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return strict_matrix(std::forward<decltype(m)>(m));
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
  template<typename F = internal::default_split_function,
    typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix horizontally.
  /// @tparam CC Coefficients for the first cut.
  /// @tparam CCs Coefficients for the second and subsequent cuts.
  template<typename CC, typename...CCs, typename Arg,
    std::enable_if_t<is_Eigen_matrix_v<Arg> and is_coefficient_v<CC>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<internal::default_split_function, CC, CCs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix horizontally.
  /// @tparam cut Number of columns in the first cut.
  /// @tparam cuts Numbers of columns in the second and subsequent cuts.
  /// @tparam F An object having a static call() method to which the result is applied before creating the tuple.
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Split a matrix diagonally and invoke function F on each segment, returning a tuple. Must be a square matrix.
  /// @tparam F An object having a static call() method to which the result is applied before creating the tuple.
  /// @tparam C Coefficients for the first cut.
  /// @tparam Cs Coefficients for each of the second and subsequent cuts.
  template<typename F, bool euclidean, typename C, typename...Cs,
    typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    constexpr auto RC_size = euclidean ? C::dimension : C::size;
    static_assert((RC_size + ... + (euclidean ? Cs::dimension : Cs::size)) <= MatrixTraits<Arg>::dimension);
    static_assert((C::size + ... + Cs::size) <= MatrixTraits<Arg>::columns);
    constexpr Eigen::Index rdim1 = RC_size, rdim2 = std::decay_t<Arg>::RowsAtCompileTime - rdim1;
    constexpr Eigen::Index cdim1 = C::size, cdim2 = std::decay_t<Arg>::RowsAtCompileTime - cdim1;
    constexpr auto g = [](auto&& m) -> decltype(auto) {
      if constexpr (std::is_lvalue_reference_v<Arg&&>)
        return std::forward<decltype(m)>(m);
      else
        return strict_matrix(std::forward<decltype(m)>(m));
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
  template<typename F = internal::default_split_function, bool euclidean = false,
    typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and not is_coefficient_v<F>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return std::tuple {};
  }

  /// Split a matrix vertically.
  /// @tparam F A class with a static <code>call</code> member to which the result is applied before creating the tuple.
  /// @tparam C Coefficients for the first cut.
  /// @tparam Cs Coefficients for each of the second and subsequent cuts.
  template<typename F, typename C, typename...Cs, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    not is_coefficient_v<F> and std::conjunction_v<is_coefficient<C>, is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, false, C, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix diagonally. Must be a square matrix.
  /// @tparam C Coefficients for the first cut.
  /// @tparam Cs Coefficients for the second and subsequent cuts.
  template<typename C, typename...Cs, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::conjunction_v<is_coefficient<C>, is_coefficient<Cs>...>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<internal::default_split_function, C, Cs...>(std::forward<Arg>(arg));
  }

  /// Split a matrix diagonally. Must be a square matrix.
  /// @tparam cut Number of rows and columns in the first cut.
  /// @tparam cuts Numbers of rows and columns in the second and subsequent cuts.
  /// @tparam F An object having a static call() method to which the result is applied before creating the tuple.
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<is_Eigen_matrix_v<Arg>, int> = 0>
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    static_assert((cut + ... + cuts) <= MatrixTraits<Arg>::columns);
    return split_diagonal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /// Get element (i, j) of matrix arg
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline auto
  get_element(const Arg& arg, const std::size_t i, const std::size_t j)
  {
    return arg.coeff(i, j);
  }


  /// Get element (i) of one-column matrix arg
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg> and MatrixTraits<Arg>::columns == 1, int> = 0>
  inline auto
  get_element(const Arg& arg, const std::size_t i)
  {
    return arg.coeff(i);
  }


  /// Set element (i, j) of matrix arg to s.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_native_Eigen_type_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
    static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit), int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i, const std::size_t j)
  {
    arg(i, j) = s;
  }


  /// Set element (i) of one-column matrix arg to s.
  template<typename Arg, typename Scalar,
    std::enable_if_t<is_native_Eigen_type_v<Arg> and not std::is_const_v<std::remove_reference_t<Arg>> and
      MatrixTraits<Arg>::columns == 1 and
    static_cast<bool>(std::decay_t<Arg>::Flags & Eigen::LvalueBit), int> = 0>
  inline void
  set_element(Arg& arg, const Scalar s, const std::size_t i)
  {
    arg(i) = s;
  }


  /// Return column <code>index</code> of Arg.
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    return std::forward<Arg>(arg).col(index);
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
  template<size_t index, typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg)
  {
    static_assert(index < MatrixTraits<Arg>::columns, "Column index out of range.");
    if constexpr(MatrixTraits<Arg>::columns == 1)
      return std::forward<Arg>(arg);
    else if constexpr(index == 0)
      return std::forward<Arg>(arg).template leftCols<1>();
    else if constexpr(index == MatrixTraits<Arg>::columns - 1)
      return std::forward<Arg>(arg).template rightCols<1>();
    else
      return std::forward<Arg>(arg).col(index);
  }


  namespace detail
  {
    template<bool index, std::size_t i, typename Arg, typename Function>
    inline Arg& do_one_column(Arg& arg, const Function& f)
    {
      auto c = column<i>(arg);
      if constexpr(index) f(c, i); else f(c);
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
      using ResultType = decltype(f(column(arg, 0)));
      if constexpr(is_ToEuclideanExpr_v<ResultType> or is_FromEuclideanExpr_v<ResultType>)
      {
        auto res = concatenate_horizontal(base_matrix(f(column(arg, ints)))...);
        return MatrixTraits<ResultType>::make(std::move(res));
      }
      else
      {
        return concatenate_horizontal(f(column(arg, ints))...);
      }
    };

    template<typename Arg, typename Function, std::size_t ... ints>
    inline auto cat_columnwise_index_impl(const Arg& arg, const Function& f, std::index_sequence<ints...>)
    {
      using ResultType = decltype(f(column(arg, 0), 0));
      if constexpr(is_ToEuclideanExpr_v<ResultType> or is_FromEuclideanExpr_v<ResultType>)
      {
        auto res = concatenate_horizontal(base_matrix(f(column(arg, ints), ints))...);
        return MatrixTraits<ResultType>::make(std::move(res));
      }
      else
      {
        return concatenate_horizontal(f(column(arg, ints), ints)...);
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


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_void_v<std::invoke_result_t<Function, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& >>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    return detail::do_columnwise_impl<false>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_void_v<std::invoke_result_t<Function,
    std::decay_t<decltype(column(std::declval<Arg>(), 0))>&, std::size_t>>, int> = 0>
  inline Arg&
  apply_columnwise(Arg& arg, const Function& f)
  {
    return detail::do_columnwise_impl<true>(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&& >>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    return detail::cat_columnwise_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    not std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<Arg>(), 0))>&&, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Arg& arg, const Function& f)
  {
    return detail::cat_columnwise_index_impl(arg, f, std::make_index_sequence<MatrixTraits<Arg>::columns>());
  }


  template<std::size_t count, typename Function,
    std::enable_if_t<is_Eigen_matrix_v<std::invoke_result_t<Function>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    return detail::cat_columnwise_impl(f, std::make_index_sequence<count>());
  }


  template<std::size_t count, typename Function,
    std::enable_if_t<is_Eigen_matrix_v<std::invoke_result_t<Function, std::size_t>>, int> = 0>
  inline auto
  apply_columnwise(const Function& f)
  {
    return detail::cat_columnwise_index_impl(f, std::make_index_sequence<count>());
  }


  ////

  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&>>, int> = 0>
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    for (std::size_t j = 0; j < MatrixTraits<Arg>::columns; j++)
    {
      for (std::size_t i = 0; i < MatrixTraits<Arg>::dimension; i++)
      {
        f(arg(i, j));
      }
    }
    return arg;
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_void_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>&, std::size_t, std::size_t>>, int> = 0>
  inline Arg&
  apply_coefficientwise(Arg& arg, const Function& f)
  {
    for (std::size_t j = 0; j < MatrixTraits<Arg>::columns; j++)
    {
      for (std::size_t i = 0; i < MatrixTraits<Arg>::dimension; i++)
      {
        f(arg(i, j), i, j);
      }
    }
    return arg;
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, std::decay_t<typename MatrixTraits<Arg>::Scalar>>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    return arg.unaryExpr(std::ref(f));
  }


  template<typename Arg, typename Function, std::enable_if_t<is_Eigen_matrix_v<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function,
      std::decay_t<typename MatrixTraits<Arg>::Scalar>, std::size_t, std::size_t>,
      typename MatrixTraits<Arg>::Scalar>, int> = 0>
  inline auto
  apply_coefficientwise(const Arg& arg, const Function& f)
  {
    Arg ret;
    for (std::size_t j = 0; j < MatrixTraits<Arg>::columns; j++)
    {
      for (std::size_t i = 0; i < MatrixTraits<Arg>::dimension; i++)
      {
        ret(i, j) = f(arg(i, j), i, j);
      }
    }
    return ret;
  }


  template<std::size_t rows, std::size_t columns, typename Function,
    std::enable_if_t<std::is_arithmetic_v<std::invoke_result_t<Function>>, int> = 0>
  inline auto
  apply_coefficientwise(const Function& f)
  {
    using Scalar = std::invoke_result_t<Function>;
    using Mat = Eigen::Matrix<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)>;
    return Mat::NullaryExpr(f);
  }


  template<std::size_t rows, std::size_t columns, typename Function,
    std::enable_if_t<std::is_arithmetic_v<std::invoke_result_t<Function, std::size_t, std::size_t>>, int> = 0>
  inline auto
  apply_coefficientwise(const Function& f)
  {
    using Scalar = std::invoke_result_t<Function, std::size_t, std::size_t>;
    Eigen::Matrix<Scalar, static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)> ret;
    for (std::size_t j = 0; j < columns; j++)
    {
      for (std::size_t i = 0; i < rows; i++)
      {
        ret(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) = f(i, j);
      }
    }
    return ret;
  }


  namespace detail
  {
    template<typename Scalar, template<typename> typename distribution_type, typename random_number_engine>
    static auto
    get_rnd(const typename distribution_type<Scalar>::param_type params)
    {
      static std::random_device rd;
      static random_number_engine rng {rd()};
      static distribution_type<Scalar> dist;
      return dist(rng, params);
    }
  }


  /**
   * Fill an Eigen matrix with random values selected from a random distribution.
   * The Gaussian distribution has mean zero and a scalar standard deviation sigma (== 1, if not specified).
   **/
  template<
    typename ReturnType,
    template<typename> typename distribution_type = std::normal_distribution,
    typename random_number_engine = std::mt19937,
    typename...Params,
    std::enable_if_t<is_Eigen_matrix_v<ReturnType>, int> = 0>
  inline auto
  randomize(Params...params)
  {
    using Scalar = typename MatrixTraits<ReturnType>::Scalar;
    using Ps = typename distribution_type<Scalar>::param_type;
    static_assert(std::is_constructible_v<Ps, Params...>,
      "Parameters params... must be constructor arguments of distribution_type<RealType>::param_type.");
    auto ps = Ps {params...};
    return strict(ReturnType::NullaryExpr([&](auto) {
      return detail::get_rnd<Scalar, distribution_type, random_number_engine>(ps);
    }));
  }


}

#endif //OPENKALMAN_EIGENMATRIXOVERLOADS_H
