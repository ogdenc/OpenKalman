/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENZERO_HPP
#define OPENKALMAN_EIGENZERO_HPP

#include <type_traits>

namespace OpenKalman::Eigen3
{
  template<typename ArgType>
  struct EigenZero : std::decay_t<ArgType>::ConstantReturnType
  {
    using ConstantReturnType = EigenZero;
    using BaseMatrix = std::decay_t<ArgType>;
    using Base = typename BaseMatrix::ConstantReturnType;

    EigenZero() : Base(BaseMatrix::Zero()) {};

    template<typename Arg>
    EigenZero(const Arg&) : EigenZero() {};

    auto operator()(std::size_t i, std::size_t j) const { return OpenKalman::internal::ElementSetter(*this, i, j); }

    auto operator()(std::size_t i) const { return OpenKalman::internal::ElementSetter(*this, i); }

    auto operator[](std::size_t i) const { return OpenKalman::internal::ElementSetter(*this, i); }
  };

} // OpenKalman::Eigen3


namespace OpenKalman
{
  //////////////
  //  Traits  //
  //////////////

  template<typename V>
  struct MatrixTraits<Eigen3::EigenZero<V>>
    : MatrixTraits<typename std::decay_t<V>::ConstantReturnType>
  {
  protected:
    using Base = MatrixTraits<typename std::decay_t<V>::ConstantReturnType>;
    using Matrix = Eigen3::EigenZero<V>;

  public:
    using BaseMatrix = V;
    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::EigenMatrixBase<Derived, Matrix>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::EigenCovarianceBase<Derived, Matrix>;

    using Strict = Eigen3::EigenZero<V>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointBaseType = Eigen3::EigenSelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularBaseType = Eigen3::EigenTriangularMatrix<Matrix, triangle_type>;
  };

} // namesace OpenKalman


namespace OpenKalman::Eigen3
{
  /////////////////
  //  Overloads  //
  /////////////////

  /// Convert to strict version of the matrix.
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  strict(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  inline auto
  to_diagonal(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::columns == 1);
    constexpr auto dim = MatrixTraits<Arg>::dimension;
    using B = strict_matrix_t<Arg, dim, dim>;
    return Eigen3::EigenZero<B>();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  transpose(Arg&& arg) noexcept
  {
    constexpr auto rows = MatrixTraits<Arg>::dimension;
    constexpr auto cols = MatrixTraits<Arg>::columns;
    using B = strict_matrix_t<Arg, cols, rows>;
    return Eigen3::EigenZero<B>();
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  adjoint(Arg&& arg) noexcept
  {
    return transpose(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  reduce_columns(Arg&& arg) noexcept
  {
    if constexpr(MatrixTraits<Arg>::columns == 1)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      using Scalar = typename MatrixTraits<Arg>::Scalar;
      return Eigen3::EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
    }
  }


  /**
   * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
   * Returns L as a lower-triangular matrix.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::is_EigenZero_v<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    constexpr auto dim = MatrixTraits<A>::dimension;
    using B = strict_matrix_t<A, dim, dim>;
    return Eigen3::EigenZero<B>();
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr A>
#else
  template<typename A, std::enable_if_t<Eigen3::is_EigenZero_v<A>, int> = 0>
#endif
  inline auto
  QR_decomposition(A&& a)
  {
    constexpr auto dim = MatrixTraits<A>::columns;
    using B = strict_matrix_t<A, dim, dim>;
    return Eigen3::EigenZero<B>();
  }


  /// Get an element of a EigenZero matrix. Always 0.
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t, const std::size_t)
  {
    return 0;
  }


  /// Get an element of a one-column EigenZero matrix. Always 0.
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg> requires (MatrixTraits<Arg>::columns == 1)
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg> and MatrixTraits<Arg>::columns == 1, int> = 0>
#endif
  constexpr auto
  get_element(const Arg&, const std::size_t)
  {
    return 0;
  }


  /// Return column <code>index</code> of Arg.
#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    return Eigen3::EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
#ifdef __cpp_concepts
  template<size_t index, Eigen3::eigen_zero_expr Arg>
#else
  template<size_t index, typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  inline decltype(auto)
  column(Arg&& arg)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr(MatrixTraits<Arg>::columns == 1)
      return std::forward<Arg>(arg);
    else
      return Eigen3::EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (is_zero_v<Arg1> and Eigen3::eigen_matrix<Arg2>) or (Eigen3::eigen_matrix<Arg1> and is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_zero_v<Arg1> and Eigen3::is_Eigen_matrix_v<Arg2>) or
      (Eigen3::is_Eigen_matrix_v<Arg1> and is_zero_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_zero_v<Arg1> and is_zero_v<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(Eigen3::eigen_zero_expr<Arg1>)
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
  requires (is_zero_v<Arg1> and Eigen3::eigen_matrix<Arg2>) or (Eigen3::eigen_matrix<Arg1> and is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_zero_v<Arg1> and Eigen3::is_Eigen_matrix_v<Arg2>) or
      (Eigen3::is_Eigen_matrix_v<Arg1> and is_zero_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(is_zero_v<Arg1> and is_zero_v<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(is_zero_v<Arg1>)
    {
      if constexpr(is_identity_v<Arg2>)
      {
        using D = typename MatrixTraits<Arg2>::template DiagonalBaseType<>;
        constexpr auto dim = MatrixTraits<Arg2>::dimension;
        using B = strict_matrix_t<D, dim, 1>;
        return MatrixTraits<D>::make(B::Constant(-1));
      }
      else
      {
        return -std::forward<Arg2>(arg2);
      }
    }
    else
    {
      return std::forward<Arg1>(arg1);
    }
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2>
  requires (is_zero_v<Arg1> and Eigen3::eigen_matrix<Arg2>) or (Eigen3::eigen_matrix<Arg1> and is_zero_v<Arg2>)
#else
  template<typename Arg1, typename Arg2,
    std::enable_if_t<(is_zero_v<Arg1> and Eigen3::is_Eigen_matrix_v<Arg2>) or
      (Eigen3::is_Eigen_matrix_v<Arg1> and is_zero_v<Arg2>), int> = 0>
#endif
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    return Eigen3::EigenZero<Eigen::Matrix<Scalar, rows, cols>>();
  }


#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires (Eigen3::eigen_zero_expr<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and Eigen3::eigen_zero_expr<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<(Eigen3::is_EigenZero_v<Arg1> and std::is_arithmetic_v<Arg2>) or
    (std::is_arithmetic_v<Arg1> and Eigen3::is_EigenZero_v<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(Eigen3::eigen_zero_expr<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<Eigen3::is_EigenZero_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


}

#endif //OPENKALMAN_EIGENZERO_HPP
