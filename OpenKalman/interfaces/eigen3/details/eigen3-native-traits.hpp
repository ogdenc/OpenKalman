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
 * \internal
 * \file
 * \brief Native Eigen3 traits for Eigen3 extensions
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


  template<typename Scalar_, auto constant, std::size_t rows, std::size_t cols>
  struct traits<OpenKalman::Eigen3::ConstantMatrix<Scalar_, constant, rows, cols>>
  {
    using StorageKind = Eigen::Dense;
    using XprKind = Eigen::MatrixXpr;
    using StorageIndex = Eigen::Index;
    using Scalar = Scalar_;
    enum
    {
      Flags = NoPreferredStorageOrderBit | EvalBeforeNestingBit | LinearAccessBit |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      RowsAtCompileTime = (rows == 0 ? Eigen::Dynamic : (Eigen::Index) rows),
      MaxRowsAtCompileTime = RowsAtCompileTime,
      ColsAtCompileTime = (cols == 0 ? Eigen::Dynamic : (Eigen::Index) cols),
      MaxColsAtCompileTime = ColsAtCompileTime
    };
  };


  template<typename Scalar_, std::size_t rows, std::size_t cols>
  struct traits<OpenKalman::Eigen3::ZeroMatrix<Scalar_, rows, cols>>
  {
    using StorageKind = Eigen::Dense;
    using XprKind = Eigen::MatrixXpr;
    using StorageIndex = Eigen::Index;
    using Scalar = Scalar_;
    enum
    {
      Flags = NoPreferredStorageOrderBit | EvalBeforeNestingBit | LinearAccessBit |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
      RowsAtCompileTime = (rows == 0 ? Eigen::Dynamic : (Eigen::Index) rows),
      MaxRowsAtCompileTime = RowsAtCompileTime,
      ColsAtCompileTime = (cols == 0 ? Eigen::Dynamic : (Eigen::Index) cols),
      MaxColsAtCompileTime = ColsAtCompileTime
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


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::Eigen3::ToEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = Coeffs::axes_only ?
              NestedTraits::Flags :
              NestedTraits::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::MatrixTraits<ArgType>::columns == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_coefficients<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(Coeffs::euclidean_dimensions);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::Eigen3::FromEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = Coeffs::axes_only ?
              NestedTraits::Flags :
              NestedTraits::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::MatrixTraits<ArgType>::columns == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_coefficients<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(Coeffs::dimensions);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<
    OpenKalman::Eigen3::FromEuclideanExpr<Coeffs, OpenKalman::Eigen3::ToEuclideanExpr<Coeffs, ArgType>>>
      : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    using NestedTraits = traits<Nested>;
    enum
    {
      Flags = Coeffs::axes_only ?
              NestedTraits::Flags :
              NestedTraits::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::MatrixTraits<ArgType>::columns == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_coefficients<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(Coeffs::dimensions);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };

}

#endif //OPENKALMAN_EIGEN3_NATIVE_TRAITS_HPP
