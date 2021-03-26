/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
#define OPENKALMAN_EIGEN3_ZEROMATRIX_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct ZeroMatrix : Eigen::Matrix<Scalar, rows, columns>::ConstantReturnType
  {
  private:
    using EMatrix = Eigen::Matrix<Scalar, rows, columns>;
    using Base = typename EMatrix::ConstantReturnType;

  public:
    ZeroMatrix() : Base {EMatrix::Zero()} {};

    constexpr Scalar operator()(std::size_t i, std::size_t j) const { return 0; }

    constexpr Scalar operator()(std::size_t i) const { return 0; }

    constexpr Scalar operator[](std::size_t i) const { return 0; }
  };

} // OpenKalman::Eigen3


namespace OpenKalman
{
  //////////////
  //  Traits  //
  //////////////

  template<typename Scalar_, std::size_t rows, std::size_t cols>
  struct MatrixTraits<Eigen3::ZeroMatrix<Scalar_, rows, cols>>
  {
    using Scalar = Scalar_;

    static constexpr std::size_t dimension = rows;
    static constexpr std::size_t columns = cols;

  private:

    using Matrix = Eigen3::ZeroMatrix<Scalar, rows, columns>;

  public:

    template<std::size_t r = rows, std::size_t c = columns, typename S = Scalar>
    using NativeMatrixFrom = Eigen::Matrix<S, r, c>;

    using SelfContainedFrom = Matrix;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<Matrix, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, Matrix>;

    static auto zero() { return Matrix {}; }

    static auto identity() { return NativeMatrixFrom<dimension, dimension>::Identity(); }

  };

} // namespace OpenKalman


namespace OpenKalman::Eigen3
{
  /////////////////
  //  Overloads  //
  /////////////////

  /// Convert to self-contained version of the matrix.
#ifdef __cpp_concepts
  template<typename...Ts, eigen_zero_expr Arg>
#else
  template<typename...Ts, typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dim = MatrixTraits<Arg>::dimension;
    return ZeroMatrix<Scalar, dim, dim>();
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  inline auto
  diagonal_of(Arg&& arg) noexcept
  {
    return ZeroMatrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::dimension, 1> {};
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto rows = MatrixTraits<Arg>::dimension;
    constexpr auto cols = MatrixTraits<Arg>::columns;
    if constexpr (rows == cols)
      return std::forward<Arg>(arg);
    else
      return ZeroMatrix<Scalar, cols, rows>();
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return transpose(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and square_matrix<Arg>, int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr(column_vector<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      constexpr auto rows = MatrixTraits<Arg>::dimension;
      return ZeroMatrix<Scalar, rows, 1>();
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto dim = MatrixTraits<A>::dimension;
    return ZeroMatrix<Scalar, dim, dim>();
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<eigen_zero_expr<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto dim = MatrixTraits<A>::columns;
    return ZeroMatrix<Scalar, dim, dim>();
  }


  /// Get an element of a ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t, const std::size_t)
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Get an element of a one-column ZeroMatrix matrix. Always 0.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg> requires column_vector<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg> and column_vector<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t)
  {
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto rows = MatrixTraits<Arg>::dimension;
    return ZeroMatrix<Scalar, rows, 1>();
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<size_t index, eigen_zero_expr Arg>
#else
  template<size_t index, typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto rows = MatrixTraits<Arg>::dimension;
    if constexpr(column_vector<Arg>)
      return std::forward<Arg>(arg);
    else
      return ZeroMatrix<Scalar, rows, 1>();
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (zero_matrix<Arg1> and eigen_matrix<Arg2>) or (eigen_matrix<Arg1> and zero_matrix<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(zero_matrix<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and zero_matrix<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(zero_matrix<Arg1> and zero_matrix<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(eigen_zero_expr<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else
    {
      return std::forward<Arg1>(arg1);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (zero_matrix<Arg1> and eigen_matrix<Arg2>) or (eigen_matrix<Arg1> and zero_matrix<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(zero_matrix<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and zero_matrix<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    // zero_matrix<Arg1>:
    else if constexpr(identity_matrix<Arg2>)
    {
      using D = typename MatrixTraits<Arg2>::template DiagonalMatrixFrom<>;
      constexpr auto dim = MatrixTraits<Arg2>::dimension;
      using B = native_matrix_t<D, dim, 1>;
      return MatrixTraits<D>::make(B::Constant(-1));
    }
    else
    {
      return -std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (zero_matrix<Arg1> and eigen_matrix<Arg2>) or (eigen_matrix<Arg1> and zero_matrix<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(zero_matrix<Arg1> and eigen_matrix<Arg2>) or
      (eigen_matrix<Arg1> and zero_matrix<Arg2>), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    return ZeroMatrix<Scalar, rows, cols>();
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_zero_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and eigen_zero_expr<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(eigen_zero_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


}

#endif //OPENKALMAN_EIGEN3_ZEROMATRIX_HPP
