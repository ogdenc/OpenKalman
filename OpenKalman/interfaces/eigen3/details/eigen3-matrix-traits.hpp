/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  //----------------------------------------------------------
  //                      MatrixBase:
  //                MatrixTraits<MatrixBase>
  //----------------------------------------------------------

  /**
   * Default matrix traits for any matrix derived from Eigen::MatrixBase.
   * @tparam M The matrix.
   */
#ifdef __cpp_concepts
  template<typename M> requires std::same_as<M, std::decay_t<M>> and
    std::derived_from<std::decay_t<M>, Eigen::MatrixBase<std::decay_t<M>>>
  struct MatrixTraits<M>
#else
  template<typename M>
  struct MatrixTraits<M, std::enable_if_t<std::is_same_v<M, std::decay_t<M>> and Eigen3::is_eigen_native_v<M>>>
#endif
  {
    using BaseMatrix = std::decay_t<M>;
    using Scalar = typename BaseMatrix::Scalar;

    static constexpr std::size_t dimension = BaseMatrix::RowsAtCompileTime;
    static constexpr std::size_t columns = BaseMatrix::ColsAtCompileTime; ///@TODO: make columns potentially dynamic (0 = dynamic?)
    //Note: rows or columns at compile time are -1 if the matrix is dynamic:
    static_assert(dimension > 0);
    static_assert(columns > 0);

    template<typename Derived>
    using MatrixBaseType = Eigen3::internal::Eigen3MatrixBase<Derived, BaseMatrix>;

    template<typename Derived>
    using CovarianceBaseType = Eigen3::internal::Eigen3CovarianceBase<Derived, BaseMatrix>;

    template<std::size_t rows = dimension, std::size_t cols = columns, typename S = Scalar>
    using StrictMatrix = Eigen::Matrix<S, (Eigen::Index) rows, (Eigen::Index) cols>;

    using Strict = typename MatrixTraits<BaseMatrix>::template StrictMatrix<>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using SelfAdjointBaseType = Eigen3::SelfAdjointMatrix<StrictMatrix<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = dimension, typename S = Scalar>
    using TriangularBaseType = Eigen3::TriangularMatrix<StrictMatrix<dim, dim, S>, triangle_type>;

    template<std::size_t dim = dimension, typename S = Scalar>
    using DiagonalBaseType = Eigen3::DiagonalMatrix<StrictMatrix<dim, 1, S>>;

#ifdef __cpp_concepts
    template<typename Arg> requires (not std::convertible_to<Arg, const Scalar>)
#else
    template<typename Arg, std::enable_if_t<not std::is_convertible_v<Arg, const Scalar>, int> = 0>
#endif
    static decltype(auto)
    make(Arg&& arg) noexcept
    {
      return std::forward<Arg>(arg);
    }

    /// Make matrix from a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<std::convertible_to<const Scalar> Arg, std::convertible_to<const Scalar> ... Args>
    requires (1 + sizeof...(Args) == dimension * columns)
#else
    template<typename Arg, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...> and
      1 + sizeof...(Args) == dimension * columns, int> = 0>
#endif
    static auto
    make(const Arg arg, const Args ... args)
    {
      return ((StrictMatrix<dimension, columns>() << arg), ... , args).finished();
    }

    static auto zero() { return Eigen3::ZeroMatrix<StrictMatrix<dimension, columns>>(); }

    static auto identity() { return StrictMatrix<dimension, dimension>::Identity(); }

  };


  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_strict_matrix<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>> : std::true_type {};


  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct is_strict<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : std::bool_constant<not (Eigen::internal::traits<XprType>::Flags & Eigen::NestByRefBit) and is_strict_v<XprType>> {};

  template<typename Scalar, typename PlainObjectType>
  struct is_strict<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, PlainObjectType>>
    : std::true_type {};

  template<typename Scalar, typename PlainObjectType>
  struct is_strict<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::true_type {};

  template<typename Scalar, typename PacketType, typename PlainObjectType>
  struct is_strict<Eigen::CwiseNullaryOp<Eigen::internal::linspaced_op<Scalar, PacketType>, PlainObjectType>>
    : std::true_type {};

  template<typename UnaryOp, typename XprType>
  struct is_strict<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<XprType>::Flags & Eigen::NestByRefBit) and is_strict_v<XprType>> {};

  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct is_strict<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<LhsType>::Flags & Eigen::NestByRefBit) and is_strict_v<LhsType> and
      not (Eigen::internal::traits<RhsType>::Flags & Eigen::NestByRefBit) and is_strict_v<RhsType>> {};

  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct is_strict<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<Arg1>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg1> and
      not (Eigen::internal::traits<Arg2>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg2> and
      not (Eigen::internal::traits<Arg3>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg3>> {};

  template<typename MatrixType, int DiagIndex>
  struct is_strict<Eigen::Diagonal<MatrixType, DiagIndex>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_strict<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::true_type {};

  template<typename DiagVectorType>
  struct is_strict<Eigen::DiagonalWrapper<DiagVectorType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<DiagVectorType>::Flags & Eigen::NestByRefBit) and is_strict_v<DiagVectorType>> {};

  template<typename XprType>
  struct is_strict<Eigen::Inverse<XprType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<XprType>::Flags & Eigen::NestByRefBit) and is_strict_v<XprType>> {};

  template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
  struct is_strict<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
    : std::true_type {};

  template<typename XprType>
  struct is_strict<Eigen::MatrixWrapper<XprType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<XprType>::Flags & Eigen::NestByRefBit) and is_strict_v<XprType>> {};

  template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
  struct is_strict<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
    : std::true_type {};

  template<typename IndicesType>
  struct is_strict<Eigen::PermutationWrapper<IndicesType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<IndicesType>::Flags & Eigen::NestByRefBit) and is_strict_v<IndicesType>> {};

  template<typename LhsType, typename RhsType, int Option>
  struct is_strict<Eigen::Product<LhsType, RhsType, Option>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<LhsType>::Flags & Eigen::NestByRefBit) and is_strict_v<LhsType> and
      not (Eigen::internal::traits<RhsType>::Flags & Eigen::NestByRefBit) and is_strict_v<RhsType>> {};

  template<typename MatrixType, int RowFactor, int ColFactor>
  struct is_strict<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename MatrixType, int Direction>
  struct is_strict<Eigen::Reverse<MatrixType, Direction>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename Arg1, typename Arg2, typename Arg3>
  struct is_strict<Eigen::Select<Arg1, Arg2, Arg3>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<Arg1>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg1> and
      not (Eigen::internal::traits<Arg2>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg2> and
      not (Eigen::internal::traits<Arg3>::Flags & Eigen::NestByRefBit) and is_strict_v<Arg3>> {};

  template<typename MatrixType, unsigned int UpLo>
  struct is_strict<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename MatrixType>
  struct is_strict<Eigen::Transpose<MatrixType>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename MatrixType, unsigned int Mode>
  struct is_strict<Eigen::TriangularView<MatrixType, Mode>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<MatrixType>::Flags & Eigen::NestByRefBit) and is_strict_v<MatrixType>> {};

  template<typename VectorType, int Size>
  struct is_strict<Eigen::VectorBlock<VectorType, Size>>
    : std::integral_constant<bool,
      not (Eigen::internal::traits<VectorType>::Flags & Eigen::NestByRefBit) and is_strict_v<VectorType>> {};

}

#endif //OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
