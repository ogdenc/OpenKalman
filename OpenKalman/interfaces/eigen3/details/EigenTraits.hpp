/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENTRAITS_HPP
#define OPENKALMAN_EIGENTRAITS_HPP

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
    : traits<typename OpenKalman::MatrixTraits<ArgType>::template SelfAdjointBaseType<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<ArgType>::template SelfAdjointBaseType<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::is_self_adjoint_v<ArgType> ? ~0 : ~LvalueBit),
    };
  };

  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::SquareRootCovariance<Coefficients, ArgType>>
    : traits<typename OpenKalman::MatrixTraits<ArgType>::template TriangularBaseType<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<ArgType>::template TriangularBaseType<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::is_triangular_v<ArgType> ? ~0 : ~LvalueBit),
    };
  };

  template<typename BaseMatrix, OpenKalman::TriangleType storage_triangle>
  struct traits<OpenKalman::Eigen3::EigenSelfAdjointMatrix<BaseMatrix, storage_triangle>>
    : traits<std::decay_t<BaseMatrix>>
  {
    using Nested = std::decay_t<BaseMatrix>;
    enum
    {
      Flags = traits<Nested>::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      RowsAtCompileTime = Nested::RowsAtCompileTime,
      ColsAtCompileTime = Nested::RowsAtCompileTime,
      MaxRowsAtCompileTime = Nested::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxRowsAtCompileTime
    };
  };

  template<typename BaseMatrix, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::Eigen3::EigenTriangularMatrix<BaseMatrix, triangle_type>>
    : traits<std::decay_t<BaseMatrix>>
  {
    using Nested = std::decay_t<BaseMatrix>;
    enum
    {
      Flags = traits<Nested>::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      RowsAtCompileTime = Nested::RowsAtCompileTime,
      ColsAtCompileTime = Nested::RowsAtCompileTime,
      MaxRowsAtCompileTime = Nested::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxRowsAtCompileTime
    };
  };

  template<typename ArgType>
  struct traits<OpenKalman::Eigen3::EigenDiagonal<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using StorageKind = Dense;
    using XprKind = MatrixXpr;
    using StorageIndex = typename Nested::StorageIndex;
    using Scalar = typename Nested::Scalar;
    enum
    {
      Flags = traits<Nested>::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      RowsAtCompileTime = Nested::RowsAtCompileTime,
      ColsAtCompileTime = Nested::RowsAtCompileTime,
      MaxRowsAtCompileTime = Nested::MaxRowsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxRowsAtCompileTime
    };
  };

  template<typename ArgType>
  struct traits<OpenKalman::Eigen3::EigenZero<ArgType>>
    : traits<typename std::decay_t<ArgType>::ConstantReturnType> {};

  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Eigen3::ToEuclideanExpr<Coefficients, ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    using StorageKind = Dense;
    using XprKind = MatrixXpr;
    using StorageIndex = typename Nested::StorageIndex;
    using Scalar = typename Nested::Scalar;
    enum
    {
      Flags = Coefficients::axes_only ?
        NestedTraits::Flags :
        ColMajor | (Nested::ColsAtCompileTime == 1 ? LinearAccessBit : 0),
      RowsAtCompileTime = Coefficients::dimension,
      MaxRowsAtCompileTime = Coefficients::dimension,
      ColsAtCompileTime = Nested::ColsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxColsAtCompileTime,
      InnerStrideAtCompileTime = Nested::InnerStrideAtCompileTime,
      OuterStrideAtCompileTime = Nested::OuterStrideAtCompileTime,
    };
  };

  template<typename Coefficients, typename ArgType>
  struct traits<OpenKalman::Eigen3::FromEuclideanExpr<Coefficients, ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    using StorageKind = Dense;
    using XprKind = MatrixXpr;
    using StorageIndex = typename Nested::StorageIndex;
    using Scalar = typename Nested::Scalar;
    enum
    {
      Flags = Coefficients::axes_only ?
        NestedTraits::Flags :
        ColMajor | (Nested::ColsAtCompileTime == 1 ? LinearAccessBit : 0),
      RowsAtCompileTime = Coefficients::size,
      MaxRowsAtCompileTime = Coefficients::size,
      ColsAtCompileTime = Nested::ColsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxColsAtCompileTime,
      InnerStrideAtCompileTime = Nested::InnerStrideAtCompileTime,
      OuterStrideAtCompileTime = Nested::OuterStrideAtCompileTime,
    };
  };

  template<typename Coefficients, typename ArgType>
  struct traits<
    OpenKalman::Eigen3::FromEuclideanExpr<Coefficients, OpenKalman::Eigen3::ToEuclideanExpr<Coefficients, ArgType>>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    using StorageKind = Dense;
    using XprKind = MatrixXpr;
    using StorageIndex = typename Nested::StorageIndex;
    using Scalar = typename Nested::Scalar;
    static constexpr auto count = Nested::ColsAtCompileTime;
    enum
    {
      Flags = (count == 1 ? LinearAccessBit : 0) | (Coefficients::axes_only ? NestedTraits::Flags : (unsigned int) ColMajor),
      RowsAtCompileTime = Coefficients::size,
      MaxRowsAtCompileTime = Coefficients::size,
      ColsAtCompileTime = Nested::ColsAtCompileTime,
      MaxColsAtCompileTime = Nested::MaxColsAtCompileTime,
      InnerStrideAtCompileTime = Nested::InnerStrideAtCompileTime,
      OuterStrideAtCompileTime = Nested::OuterStrideAtCompileTime,
    };
  };

}

#endif //OPENKALMAN_EIGENTRAITS_HPP
