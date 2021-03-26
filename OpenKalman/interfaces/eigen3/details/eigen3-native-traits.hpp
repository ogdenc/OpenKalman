/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_NATIVE_TRAITS_HPP
#define OPENKALMAN_EIGEN3_NATIVE_TRAITS_HPP

namespace Eigen::internal
{
  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Mean<Coefficients, ArgType>>
    : traits<std::decay_t<ArgType>>
  {
    using Base = traits<std::decay_t<ArgType>>;
    enum
    {
      Flags = (Coefficients::axes_only ? Base::Flags : Base::Flags & ~LvalueBit),
    };
  };


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::EuclideanMean<Coeffs, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Covariance<Coefficients, ArgType>>
    : traits<typename OpenKalman::MatrixTraits<ArgType>::template SelfAdjointMatrixFrom<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<ArgType>::template SelfAdjointMatrixFrom<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::self_adjoint_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
  };


  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::SquareRootCovariance<Coefficients, ArgType>>
    : traits<typename OpenKalman::MatrixTraits<ArgType>::template TriangularMatrixFrom<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<ArgType>::template TriangularMatrixFrom<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::triangular_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
  };


  template<typename NestedMatrix, OpenKalman::TriangleType storage_triangle>
  struct traits<OpenKalman::Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : traits<std::decay_t<NestedMatrix>>
  {
    using Nested = std::decay_t<NestedMatrix>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = NestedTraits::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
    };
  };


  template<typename NestedMatrix, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : traits<std::decay_t<NestedMatrix>>
  {
    using Nested = std::decay_t<NestedMatrix>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = NestedTraits::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
    };
  };


  template<typename ArgType>
  struct traits<OpenKalman::Eigen3::DiagonalMatrix<ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = NestedTraits::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      ColsAtCompileTime = NestedTraits::RowsAtCompileTime,
      MaxColsAtCompileTime = NestedTraits::MaxRowsAtCompileTime
    };
  };


  template<typename Scalar, std::size_t rows, std::size_t cols>
  struct traits<OpenKalman::Eigen3::ZeroMatrix<Scalar, rows, cols>>
    : traits<typename Matrix<Scalar, rows, cols>::ConstantReturnType> {};


  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Eigen3::ToEuclideanExpr<Coefficients, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = Coefficients::axes_only ?
        NestedTraits::Flags :
        ColMajor | (NestedTraits::ColsAtCompileTime == 1 ? LinearAccessBit : 0),
      RowsAtCompileTime = static_cast<Index>(Coefficients::dimension),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Eigen3::FromEuclideanExpr<Coefficients, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = Coefficients::axes_only ?
        NestedTraits::Flags :
        ColMajor | (NestedTraits::ColsAtCompileTime == 1 ? LinearAccessBit : 0),
      RowsAtCompileTime = static_cast<Index>(Coefficients::size),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coefficients, typename ArgType>
  struct traits<
    OpenKalman::Eigen3::FromEuclideanExpr<Coefficients, OpenKalman::Eigen3::ToEuclideanExpr<Coefficients, ArgType>>>
      : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = (NestedTraits::ColsAtCompileTime == 1 ? LinearAccessBit : 0) | (Coefficients::axes_only ?
        NestedTraits::Flags :
        (unsigned int) ColMajor),
      RowsAtCompileTime = static_cast<Index>(Coefficients::size),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };

}

#endif //OPENKALMAN_EIGEN3_NATIVE_TRAITS_HPP
