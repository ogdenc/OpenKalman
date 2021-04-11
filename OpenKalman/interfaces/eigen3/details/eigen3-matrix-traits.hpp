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
 * \file
 * \brief Concepts as applied to native Eigen3 matrix classes.
 */

#ifndef OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{

  /*
   * \internal
   * \brief Default matrix traits for any self-contained \ref eigen_native.
   * \tparam M The matrix.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_native M> requires std::same_as<M, std::decay_t<M>>
  struct MatrixTraits<M>
#else
  template<typename M>
  struct MatrixTraits<M, std::enable_if_t<Eigen3::eigen_native<M> and std::is_same_v<M, std::decay_t<M>>>>
#endif
  {
    using Scalar = typename M::Scalar;

    static constexpr std::size_t rows = M::RowsAtCompileTime;
    static constexpr std::size_t columns = M::ColsAtCompileTime; //<\todo: make columns potentially dynamic (0 = dynamic?)
    //Note: rows or columns at compile time are -1 if the matrix is dynamic:
    static_assert(rows > 0, "Cannot currently use dynamically sized matrices with OpenKalman.");
    static_assert(columns > 0, "Cannot currently use dynamically sized matrices with OpenKalman.");

    // Cannot use dimension and columns constant expressions here because of bug in GCC 10.1.0 (but not clang 10.0.0):
    template<std::size_t r = std::size_t(M::RowsAtCompileTime),
      std::size_t c = std::size_t(M::ColsAtCompileTime), typename S = Scalar>
    using NativeMatrixFrom = Eigen::Matrix<S, (Eigen::Index) r, (Eigen::Index) c>;

    using SelfContainedFrom = NativeMatrixFrom<>;

    template<TriangleType storage_triangle = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = Eigen3::SelfAdjointMatrix<NativeMatrixFrom<dim, dim, S>, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::lower, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = Eigen3::TriangularMatrix<NativeMatrixFrom<dim, dim, S>, triangle_type>;

    template<std::size_t dim = rows, typename S = Scalar>
    using DiagonalMatrixFrom = Eigen3::DiagonalMatrix<NativeMatrixFrom<dim, 1, S>>;

    template<typename Derived>
    using MatrixBaseFrom = Eigen3::internal::Eigen3MatrixBase<Derived, M>;

#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    static decltype(auto) make(Arg&& arg) noexcept
    {
      return std::forward<Arg>(arg);
    }

    // Make matrix from a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<std::convertible_to<Scalar> Arg, std::convertible_to<Scalar> ... Args>
    requires (1 + sizeof...(Args) == rows * columns)
#else
    template<typename Arg, typename ... Args,
      std::enable_if_t<std::conjunction_v<std::is_convertible<Arg, Scalar>, std::is_convertible<Args, Scalar>...> and
      1 + sizeof...(Args) == rows * columns, int> = 0>
#endif
    static auto make(const Arg arg, const Args ... args)
    {
      return ((NativeMatrixFrom<rows, columns>() << arg), ... , args).finished();
    }

    static auto zero() { return Eigen3::ZeroMatrix<Scalar, rows, columns>(); }

    static auto identity() { return NativeMatrixFrom<rows, rows>::Identity(); }

  };


  /*
   * \internal
   * \brief Matrix traits for Eigen::SelfAdjointView.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_native M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>> : MatrixTraits<M>
#else
  template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::SelfAdjointView<M, UpLo>, std::enable_if_t<Eigen3::eigen_native<M>>> : MatrixTraits<M>
#endif
  {
    using MatrixTraits<M>::rows;
    using typename MatrixTraits<M>::Scalar;
    static constexpr TriangleType storage_triangle = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

    template<TriangleType storage_triangle = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;

    template<TriangleType triangle_type = storage_triangle, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;

#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::SelfAdjointView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


  /*
   * \internal
   * \brief Matrix traits for Eigen::TriangularView.
   */
#ifdef __cpp_concepts
  template<Eigen3::eigen_native M, unsigned int UpLo>
  struct MatrixTraits<Eigen::TriangularView<M, UpLo>> : MatrixTraits<M>
#else
    template<typename M, unsigned int UpLo>
  struct MatrixTraits<Eigen::TriangularView<M, UpLo>, std::enable_if_t<Eigen3::eigen_native<M>>> : MatrixTraits<M>
#endif
  {
    using MatrixTraits<M>::rows;
    using typename MatrixTraits<M>::Scalar;
    static constexpr TriangleType triangle_type = UpLo & Eigen::Upper ? TriangleType::upper : TriangleType::lower;

    template<TriangleType storage_triangle = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using SelfAdjointMatrixFrom = typename MatrixTraits<M>::template SelfAdjointMatrixFrom<storage_triangle, dim, S>;

    template<TriangleType triangle_type = triangle_type, std::size_t dim = rows, typename S = Scalar>
    using TriangularMatrixFrom = typename MatrixTraits<M>::template TriangularMatrixFrom<triangle_type, dim, S>;

#ifdef __cpp_concepts
    template<Eigen3::eigen_native Arg>
#else
    template<typename Arg, std::enable_if_t<Eigen3::eigen_native<Arg>, int> = 0>
#endif
    auto make(Arg& arg) noexcept
    {
      return Eigen::TriangularView<std::remove_reference_t<Arg>, UpLo>(arg);
    }

  };


  namespace internal
  {

    namespace detail
    {
      // T is self-contained and Eigen stores it by value rather than by reference.
      template<typename T>
#ifdef __cpp_concepts
      concept stores =
#else
      static constexpr bool stores =
#endif
        self_contained<T> and ((Eigen::internal::traits<T>::Flags & Eigen::NestByRefBit) == 0);
    }

    ///////  is_self_contained  ///////

    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
    struct is_self_contained<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
      : std::bool_constant<detail::stores<XprType>> {};

    template<typename Scalar, typename PlainObjectType>
    struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, PlainObjectType>>
      : std::true_type {};

    template<typename Scalar, typename PlainObjectType>
    struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
      : std::true_type {};

    template<typename Scalar, typename PacketType, typename PlainObjectType>
    struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::linspaced_op<Scalar, PacketType>, PlainObjectType>>
      : std::true_type {};

    template<typename UnaryOp, typename XprType>
    struct is_self_contained<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
      : std::bool_constant<detail::stores<XprType>> {};

    template<typename BinaryOp, typename LhsType, typename RhsType>
    struct is_self_contained<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
      : std::bool_constant<detail::stores<LhsType> and detail::stores<RhsType>> {};

    template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
    struct is_self_contained<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
      : std::bool_constant<detail::stores<Arg1> and detail::stores<Arg2> and detail::stores<Arg3>> {};

    template<typename MatrixType, int DiagIndex>
    struct is_self_contained<Eigen::Diagonal<MatrixType, DiagIndex>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct is_self_contained<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : std::true_type {};

    template<typename DiagVectorType>
    struct is_self_contained<Eigen::DiagonalWrapper<DiagVectorType>>
      : std::bool_constant<detail::stores<DiagVectorType>> {};

    template<typename XprType>
    struct is_self_contained<Eigen::Inverse<XprType>>
      : std::bool_constant<detail::stores<XprType>> {};

    template<typename S, int rows, int cols, int options, int maxrows, int maxcols>
    struct is_self_contained<Eigen::Matrix<S, rows, cols, options, maxrows, maxcols>>
      : std::true_type {};

    template<typename XprType>
    struct is_self_contained<Eigen::MatrixWrapper<XprType>>
      : std::bool_constant<detail::stores<XprType>> {};

    template<int SizeAtCompileTime, int MaxSizeAtCompileTime, typename StorageIndex>
    struct is_self_contained<Eigen::PermutationMatrix<SizeAtCompileTime, MaxSizeAtCompileTime, StorageIndex>>
      : std::true_type {};

    template<typename IndicesType>
    struct is_self_contained<Eigen::PermutationWrapper<IndicesType>>
      : std::bool_constant<detail::stores<IndicesType>> {};

    template<typename LhsType, typename RhsType, int Option>
    struct is_self_contained<Eigen::Product<LhsType, RhsType, Option>>
      : std::bool_constant<detail::stores<LhsType> and detail::stores<RhsType>> {};

    template<typename MatrixType, int RowFactor, int ColFactor>
    struct is_self_contained<Eigen::Replicate<MatrixType, RowFactor, ColFactor>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename MatrixType, int Direction>
    struct is_self_contained<Eigen::Reverse<MatrixType, Direction>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename Arg1, typename Arg2, typename Arg3>
    struct is_self_contained<Eigen::Select<Arg1, Arg2, Arg3>>
      : std::bool_constant<detail::stores<Arg1> and detail::stores<Arg2> and detail::stores<Arg3>> {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_self_contained<Eigen::SelfAdjointView<MatrixType, UpLo>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename MatrixType>
    struct is_self_contained<Eigen::Transpose<MatrixType>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_self_contained<Eigen::TriangularView<MatrixType, UpLo>>
      : std::bool_constant<detail::stores<MatrixType>> {};

    template<typename VectorType, int Size>
    struct is_self_contained<Eigen::VectorBlock<VectorType, Size>>
      : std::bool_constant<detail::stores<VectorType>> {};


    ///////  is_modifiable_native  ///////

    // no_assignment_operator is a private base class of Cwise___Operator, Select, DiagonalWrapper, and a few others.
    // This also includes ZeroMatrix, which derives from CwiseNullaryOperator.
#ifdef __cpp_concepts
    template<typename T, typename U> requires std::is_base_of_v<Eigen::internal::no_assignment_operator, T>
    struct is_modifiable_native<T, U>
#else
    template<typename T, typename U>
    struct is_modifiable_native<T, U, std::enable_if_t<std::is_base_of_v<Eigen::internal::no_assignment_operator, T>>>
#endif
      : std::false_type {};


    template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel, typename U>
    struct is_modifiable_native<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>, U>
      : std::bool_constant<bool(Eigen::internal::traits<
          Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>::Flags & Eigen::LvalueBit) and
        (not has_const<XprType>::value) and
        (MatrixTraits<U>::rows == BlockRows) and (MatrixTraits<U>::columns == BlockCols) and
        (std::is_same_v<typename MatrixTraits<XprType>::Scalar, typename MatrixTraits<U>::Scalar>)> {};


    template<typename XprType, typename U>
    struct is_modifiable_native<Eigen::Inverse<XprType>, U>
      : std::false_type {};


    template<typename LhsType, typename RhsType, int Option, typename U>
    struct is_modifiable_native<Eigen::Product<LhsType, RhsType, Option>, U>
      : std::false_type {};


    template<typename MatrixType, int RowFactor, int ColFactor, typename U>
    struct is_modifiable_native<Eigen::Replicate<MatrixType, RowFactor, ColFactor>, U>
      : std::false_type {};


    template<typename MatrixType, int Direction, typename U>
    struct is_modifiable_native<Eigen::Reverse<MatrixType, Direction>, U>
      : std::false_type {};

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
