/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENZERO_H
#define OPENKALMAN_EIGENZERO_H

#include <type_traits>

namespace OpenKalman
{
  template<typename ArgType>
  struct EigenZero : std::decay_t<ArgType>::ConstantReturnType
  {
    using ConstantReturnType = EigenZero;
    using BaseMatrix = std::decay_t<ArgType>;
    using Base = typename BaseMatrix::ConstantReturnType;

    EigenZero() : Base(BaseMatrix::Zero()) {};

    EigenZero(const Base& z) : Base(z) {};

    EigenZero(Base&& z) : Base(std::move(z)) {};
  };


  //////////////
  //  Traits  //
  //////////////

  template<typename V>
  struct MatrixTraits<OpenKalman::EigenZero<V>>
    : MatrixTraits<typename std::decay_t<V>::ConstantReturnType>
  {
  protected:
    using Base = MatrixTraits<typename std::decay_t<V>::ConstantReturnType>;
    using Matrix = OpenKalman::EigenZero<V>;

  public:
    using BaseMatrix = V;
    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, Matrix>;

    template<typename Derived>
    using CovarianceBaseType = internal::EigenCovarianceBase<Derived, Matrix>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularBaseType = EigenTriangularMatrix<Matrix, triangle_type>;
  };


  /////////////////
  //  Overloads  //
  /////////////////

  /// Convert to strict version of the matrix.
  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<OpenKalman::is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr auto
  determinant(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr auto
  trace(Arg&& arg) noexcept
  {
    static_assert(MatrixTraits<Arg>::dimension == MatrixTraits<Arg>::columns);
    return static_cast<typename MatrixTraits<Arg>::Scalar>(0);
  }


  /// Create a column vector by taking the mean of each row in a set of column vectors.
  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
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
      return EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
    }
  }


  /// Return column <code>index</code> of Arg.
  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  inline auto
  column(Arg&& arg, const std::size_t index)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    return EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
  }


  /// Return column <code>index</code> of Arg. Constexpr index version.
  template<size_t index, typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  inline decltype(auto)
  column(Arg&& arg)
  {
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    if constexpr(MatrixTraits<Arg>::columns == 1)
      return std::forward<Arg>(arg);
    else
      return EigenZero<Eigen::Matrix<Scalar, MatrixTraits<Arg>::dimension, 1>>();
  }


  ///////////////////////////////
  //        Arithmetic         //
  ///////////////////////////////

  template<typename Arg1, typename Arg2,
    std::enable_if_t<(OpenKalman::is_zero_v<Arg1> and OpenKalman::is_Eigen_matrix_v<Arg2>) or
      (OpenKalman::is_Eigen_matrix_v<Arg1> and OpenKalman::is_zero_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_zero_v<Arg1> and OpenKalman::is_zero_v<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(OpenKalman::is_EigenZero_v<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else
    {
      return std::forward<Arg1>(arg1);
    }
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(OpenKalman::is_zero_v<Arg1> and OpenKalman::is_Eigen_matrix_v<Arg2>) or
      (OpenKalman::is_Eigen_matrix_v<Arg1> and OpenKalman::is_zero_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    static_assert(MatrixTraits<Arg1>::dimension == MatrixTraits<Arg2>::dimension);
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::columns);
    if constexpr(OpenKalman::is_zero_v<Arg1> and OpenKalman::is_zero_v<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(OpenKalman::is_zero_v<Arg1>)
    {
      if constexpr(OpenKalman::is_identity_v<Arg2>)
      {
        using D = typename MatrixTraits<Arg2>::template DiagonalBaseType<>;
        constexpr auto dim = MatrixTraits<Arg2>::dimension;
        using B = typename MatrixTraits<D>::template StrictMatrix<dim, 1>;
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


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(OpenKalman::is_zero_v<Arg1> and OpenKalman::is_Eigen_matrix_v<Arg2>) or
      (OpenKalman::is_Eigen_matrix_v<Arg1> and OpenKalman::is_zero_v<Arg2>), int> = 0>
  inline auto operator*(const Arg1& arg1, const Arg2& arg2)
  {
    static_assert(MatrixTraits<Arg1>::columns == MatrixTraits<Arg2>::dimension);
    using Scalar = typename MatrixTraits<Arg1>::Scalar;
    constexpr auto rows = MatrixTraits<Arg1>::dimension;
    constexpr auto cols = MatrixTraits<Arg2>::columns;
    return EigenZero<Eigen::Matrix<Scalar, rows, cols>>();
  }


  template<typename Arg1, typename Arg2,
    std::enable_if_t<(OpenKalman::is_EigenZero_v<Arg1> and std::is_arithmetic_v<Arg2>) or
      (std::is_arithmetic_v<Arg1> and OpenKalman::is_EigenZero_v<Arg2>), int> = 0>
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr(OpenKalman::is_EigenZero_v<Arg1>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      return std::forward<Arg2>(arg2);
    }
  }


  template<typename Arg,
    std::enable_if_t<OpenKalman::is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto) operator-(Arg&& arg)
  {
    return std::forward<Arg>(arg);
  }


}

#endif //OPENKALMAN_EIGENZERO_H
