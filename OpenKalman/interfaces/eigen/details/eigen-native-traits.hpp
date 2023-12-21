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

#ifndef OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP
#define OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP

namespace Eigen::internal
{
  template<typename NestedMatrix>
  struct traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix>>
    : traits<std::decay_t<NestedMatrix>>
  {
    enum
    {
      RowsAtCompileTime = 1,
      ColsAtCompileTime = 1,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
    };
  };


  template<typename NestedMatrix, typename Rows>
  struct traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows>>
    : traits<std::decay_t<NestedMatrix>>
  {
  private:

    static constexpr auto rows = OpenKalman::fixed_vector_space_descriptor<Rows> ? static_cast<int>(OpenKalman::dimension_size_of_v<Rows>) : Eigen::Dynamic;
    static constexpr auto row_major_bit = rows != 1 ? 0x0 : (traits<std::decay_t<NestedMatrix>>::Flags & RowMajorBit);

  public:

    enum
    {
      RowsAtCompileTime = rows,
      ColsAtCompileTime = 1,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
      Flags = (traits<std::decay_t<NestedMatrix>>::Flags & ~RowMajorBit) | row_major_bit,
    };
  };


  template<typename NestedMatrix, typename Rows, typename Cols>
  struct traits<OpenKalman::internal::FixedSizeAdapter<NestedMatrix, Rows, Cols>>
    : traits<std::decay_t<NestedMatrix>>
  {
  private:

    static constexpr auto rows = OpenKalman::fixed_vector_space_descriptor<Rows> ? static_cast<int>(OpenKalman::dimension_size_of_v<Rows>) : Eigen::Dynamic;
    static constexpr auto cols = OpenKalman::fixed_vector_space_descriptor<Cols> ? static_cast<int>(OpenKalman::dimension_size_of_v<Cols>) : Eigen::Dynamic;
    static constexpr auto row_major_bit =
      rows == 1 and cols != 1 ? RowMajorBit :
      rows != 1 and cols == 1 ? 0x0 :
      (traits<std::decay_t<NestedMatrix>>::Flags & RowMajorBit);

  public:

    enum
    {
      RowsAtCompileTime = rows,
      ColsAtCompileTime = cols,
      MaxRowsAtCompileTime = RowsAtCompileTime,
      MaxColsAtCompileTime = ColsAtCompileTime,
      Flags = (traits<std::decay_t<NestedMatrix>>::Flags & ~RowMajorBit) | row_major_bit,
    };
  };


  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct traits<OpenKalman::ConstantAdapter<PatternMatrix, Scalar, constant...>> : traits<std::decay_t<PatternMatrix>>
  {
    using StorageKind = Eigen::Dense;
    using B = traits<std::decay_t<PatternMatrix>>;
    using M = Matrix<Scalar, B::RowsAtCompileTime, B::ColsAtCompileTime>;
    enum
    {
      Flags = NoPreferredStorageOrderBit | LinearAccessBit | (traits<M>::Flags & RowMajorBit) |
        (packet_traits<Scalar>::Vectorizable ? PacketAccessBit : 0),
    };
  };


  template<typename NestedMatrix, OpenKalman::HermitianAdapterType storage_triangle>
  struct traits<OpenKalman::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : traits<std::decay_t<NestedMatrix>>
  {
    using Base = traits<std::decay_t<NestedMatrix>>;
    enum
    {
      Flags = Base::Flags &
        ~DirectAccessBit &
        ~(OpenKalman::one_dimensional<NestedMatrix> ? 0x0 : LinearAccessBit | PacketAccessBit) &
        ~(OpenKalman::complex_number<typename Base::Scalar> ? LvalueBit : 0x0),
    };
  };


  template<typename NestedMatrix, OpenKalman::TriangleType triangle_type>
  struct traits<OpenKalman::TriangularMatrix<NestedMatrix, triangle_type>>
    : traits<std::decay_t<NestedMatrix>>
  {
    static constexpr auto BaseFlags = traits<std::decay_t<NestedMatrix>>::Flags;
    enum
    {
      Flags = BaseFlags &
        ~DirectAccessBit &
        ~(OpenKalman::one_dimensional<NestedMatrix> ? 0x0 : LinearAccessBit | PacketAccessBit),
    };
  };


  template<typename ArgType>
  struct traits<OpenKalman::DiagonalMatrix<ArgType>> : traits<std::decay_t<ArgType>>
  {
    using Base = traits<std::decay_t<ArgType>>;
    enum
    {
      Flags = Base::Flags &
        ~DirectAccessBit &
        ~(OpenKalman::one_dimensional<ArgType> ? 0x0 : LinearAccessBit | PacketAccessBit),
      ColsAtCompileTime = Base::RowsAtCompileTime,
      MaxColsAtCompileTime = Base::MaxRowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::ToEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    static constexpr auto BaseFlags = traits<std::decay_t<ArgType>>::Flags;
    enum
    {
      Flags = OpenKalman::euclidean_vector_space_descriptor<Coeffs> ? BaseFlags :
        BaseFlags & ~DirectAccessBit & ~PacketAccessBit & ~LvalueBit &
          ~(OpenKalman::vector<ArgType> ? 0 : LinearAccessBit),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_vector_space_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(OpenKalman::euclidean_dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::FromEuclideanExpr<Coeffs, ArgType>> : traits<std::decay_t<ArgType>>
  {
    static constexpr auto BaseFlags = traits<std::decay_t<ArgType>>::Flags;
    enum
    {
      Flags = OpenKalman::euclidean_vector_space_descriptor<Coeffs> ? BaseFlags :
        BaseFlags & ~DirectAccessBit & ~PacketAccessBit & ~LvalueBit &
          ~(OpenKalman::vector<ArgType> ? 0 : LinearAccessBit),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_vector_space_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(OpenKalman::dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename Coeffs, typename ArgType>
  struct traits<
    OpenKalman::FromEuclideanExpr<Coeffs, OpenKalman::ToEuclideanExpr<Coeffs, ArgType>>> : traits<std::decay_t<ArgType>>
  {
    static constexpr auto BaseFlags = traits<std::decay_t<ArgType>>::Flags;
    enum
    {
      Flags = OpenKalman::euclidean_vector_space_descriptor<Coeffs> ? BaseFlags :
        BaseFlags & ~DirectAccessBit & ~PacketAccessBit & ~LvalueBit &
          ~(OpenKalman::vector<ArgType> ? 0 : LinearAccessBit),
      RowsAtCompileTime = [] {
        if constexpr (OpenKalman::dynamic_vector_space_descriptor<Coeffs>) return Eigen::Dynamic;
        else return static_cast<Index>(OpenKalman::dimension_size_of_v<Coeffs>);
      }(),
      MaxRowsAtCompileTime = RowsAtCompileTime,
    };
  };


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::Mean<TypedIndex, ArgType>> : traits<std::decay_t<ArgType>>
  {
    static constexpr auto BaseFlags = traits<std::decay_t<ArgType>>::Flags;
    enum
    {
      Flags = OpenKalman::euclidean_vector_space_descriptor<TypedIndex> ? BaseFlags :
        BaseFlags & ~DirectAccessBit & ~PacketAccessBit & ~LvalueBit,
    };
  };


  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
  struct traits<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, ArgType>> : traits<std::decay_t<ArgType>> {};


  template<typename Coeffs, typename ArgType>
  struct traits<OpenKalman::EuclideanMean<Coeffs, ArgType>> : traits<std::decay_t<ArgType>> {};


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::Covariance<TypedIndex, ArgType>>
    : traits<std::decay_t<std::conditional_t<OpenKalman::hermitian_matrix<ArgType>, ArgType, decltype(cholesky_square(std::declval<ArgType>()))>>>
  {
    using Base = traits<std::decay_t<std::conditional_t<OpenKalman::hermitian_matrix<ArgType>, ArgType, decltype(cholesky_square(std::declval<ArgType>()))>>>;
    enum
    {
      Flags = Base::Flags & ~(OpenKalman::hermitian_matrix<ArgType> ? 0x0 : DirectAccessBit | PacketAccessBit | LvalueBit),
    };
  };


  template<typename TypedIndex, typename ArgType>
  struct traits<OpenKalman::SquareRootCovariance<TypedIndex, ArgType>>
    : traits<std::decay_t<std::conditional_t<OpenKalman::triangular_matrix<ArgType>, ArgType, decltype(cholesky_factor(std::declval<ArgType>()))>>>
  {
    using Base = traits<std::decay_t<std::conditional_t<OpenKalman::triangular_matrix<ArgType>, ArgType, decltype(cholesky_factor(std::declval<ArgType>()))>>>;
    enum
    {
      Flags = Base::Flags & ~(OpenKalman::triangular_matrix<ArgType> ? 0x0 : DirectAccessBit | PacketAccessBit | LvalueBit),
    };
  };


}

#endif //OPENKALMAN_EIGEN_NATIVE_TRAITS_HPP
