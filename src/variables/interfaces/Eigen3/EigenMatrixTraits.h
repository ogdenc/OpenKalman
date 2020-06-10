/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENMATRIXTRAITS_H
#define OPENKALMAN_EIGENMATRIXTRAITS_H

#include <random>
#include <type_traits>

namespace OpenKalman
{
  //----------------------------------------------------------
  //                      MatrixBase:
  //                MatrixTraits<MatrixBase>
  //----------------------------------------------------------

  template<typename Matrix>
  struct MatrixTraits<Matrix,
    std::enable_if_t<std::is_same_v<Matrix, std::decay_t<Matrix>> and OpenKalman::is_native_Eigen_type_v<Matrix>>>
  {
    using BaseMatrix = Matrix;
    using Scalar = typename Matrix::Scalar;
    using Index = Eigen::Index;

    static constexpr std::size_t dimension = Matrix::RowsAtCompileTime;
    static constexpr std::size_t columns = Matrix::ColsAtCompileTime; ///@TODO: make columns potentially dynamic (0 = dynamic?)
    //Note: rows or columns at compile time are -1 if the matrix is dynamic:
    static_assert(dimension > 0);
    static_assert(columns > 0);

    template<typename Derived>
    using MatrixBaseType = internal::EigenMatrixBase<Derived, Matrix>;

    template<typename Derived>
    using CovarianceBaseType = internal::EigenCovarianceBase<Derived, Matrix>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = Eigen::Matrix<S, (Index) rows, (Index) cols>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = EigenSelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = EigenTriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = EigenDiagonal<StrictMatrix<dim, 1, S>>;

    /// Make matrix from a list of coefficients in row-major order.
    template<typename Arg, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...>, int> = 0>
    static auto
    make(const Arg arg, const Args ... args)
    {
      static_assert(1 + sizeof...(Args) == dimension * columns);
      return ((StrictMatrix<>() << arg), ... , args).finished();
    }

    static auto zero() { return EigenZero<StrictMatrix<>>(); }

    static auto identity() { return StrictMatrix<dimension, dimension, Scalar>::Identity(); }

  };


  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_strict_matrix<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>> : std::true_type {};

  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_strict<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>> : std::true_type {};


}

#endif //OPENKALMAN_EIGENMATRIXTRAITS_H
