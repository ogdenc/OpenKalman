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
  using namespace OpenKalman;


  template<typename NestedMatrix>
  struct traits<OpenKalman::Eigen3::EigenWrapper<NestedMatrix>> : traits<NestedMatrix> {};


  template<typename PatternMatrix, auto constant>
  struct traits<OpenKalman::ConstantAdapter<PatternMatrix, constant>> : traits<std::decay_t<PatternMatrix>>
  {
    using StorageKind = Eigen::Dense;
    using B = traits<std::decay_t<PatternMatrix>>;
    using Scalar = typename B::Scalar;
    using M = Matrix<Scalar, B::RowsAtCompileTime, B::ColsAtCompileTime>;
    enum
    {
      Flags = NoPreferredStorageOrderBit | LinearAccessBit | (traits<M>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
    };
  };


  template<typename NestedMatrix, OpenKalman::TriangleType storage_triangle>
  struct traits<OpenKalman::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : traits<std::decay_t<NestedMatrix>>
  {
    using Nested = std::decay_t<NestedMatrix>;
    using Scalar = typename traits<Nested>::Scalar;
    enum
    {
      Flags = traits<Nested>::Flags &
        (~DirectAccessBit) &
        (~(storage_triangle == TriangleType::diagonal or one_by_one_matrix<NestedMatrix> ? 0 : LinearAccessBit)) &
        (~PacketAccessBit) &
        ((storage_triangle == TriangleType::diagonal and not one_by_one_matrix<NestedMatrix>) or
          OpenKalman::complex_number<Scalar> ? ~LvalueBit : ~static_cast<decltype(LvalueBit)>(0u)),
    };
  };


  template<typename NestedMatrix, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::TriangularMatrix<NestedMatrix, triangle_type>>
    : traits<std::decay_t<NestedMatrix>>
  {
    using Nested = std::decay_t<NestedMatrix>;
    enum
    {
      Flags = traits<Nested>::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
    };
  };


  template<typename ArgType>
  struct traits<OpenKalman::DiagonalMatrix<ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    enum
    {
      Flags = traits<Nested>::Flags & (~DirectAccessBit) & (~LinearAccessBit) & (~PacketAccessBit),
      ColsAtCompileTime = traits<Nested>::RowsAtCompileTime,
      MaxColsAtCompileTime = traits<Nested>::MaxRowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::ToEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    enum
    {
      Flags = euclidean_index_descriptor<Coeffs> ?
              traits<Nested>::Flags :
              traits<Nested>::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::column_dimension_of_v<ArgType> == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_index_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(euclidean_dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::FromEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    enum
    {
      Flags = euclidean_index_descriptor<Coeffs> ?
              traits<Nested>::Flags :
              traits<Nested>::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::column_dimension_of_v<ArgType> == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_index_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<
    OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>>
      : traits<std::decay_t<ArgType>>
  {
    using Nested = std::decay_t<ArgType>;
    enum
    {
      Flags = euclidean_index_descriptor<Coeffs> ?
              traits<Nested>::Flags :
              traits<Nested>::Flags & (~DirectAccessBit) & (~PacketAccessBit) & (~LvalueBit) &
                (~(OpenKalman::column_dimension_of_v<ArgType> == 1 ? 0 : LinearAccessBit)),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_index_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::Mean<TypedIndex, ArgType>>
    : traits<std::decay_t<ArgType>>
  {
    using Base = traits<std::decay_t<ArgType>>;
    enum
    {
      Flags = (euclidean_index_descriptor<TypedIndex> ? Base::Flags : Base::Flags & ~LvalueBit),
    };
  };


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::EuclideanMean<Coeffs, ArgType>>
    : traits<std::decay_t<ArgType>> {};


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::Covariance<TypedIndex, ArgType>>
    : traits<typename OpenKalman::MatrixTraits<std::decay_t<ArgType>>::template SelfAdjointMatrixFrom<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<std::decay_t<ArgType>>::template SelfAdjointMatrixFrom<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::hermitian_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
  };


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::SquareRootCovariance<TypedIndex, ArgType>>
    : traits<typename OpenKalman::MatrixTraits<std::decay_t<ArgType>>::template TriangularMatrixFrom<>>
  {
    using Base = traits<typename OpenKalman::MatrixTraits<std::decay_t<ArgType>>::template TriangularMatrixFrom<>>;
    enum
    {
      Flags = Base::Flags & (OpenKalman::triangular_matrix<ArgType> ? ~0 : ~LvalueBit),
    };
  };


}

#endif //OPENKALMAN_EIGEN3_NATIVE_TRAITS_HPP
