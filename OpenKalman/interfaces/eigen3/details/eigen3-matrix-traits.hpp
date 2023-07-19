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


namespace OpenKalman::Eigen3
{
  // ------------------------------------------------------------------------------------------- //
  //   eigen_SelfAdjointView, eigen_TriangularView, eigen_DiagonalMatrix, eigen_DiagonalWrapper  //
  // ------------------------------------------------------------------------------------------- //

  namespace detail
  {
    template<typename T>
    struct is_eigen_SelfAdjointView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_SelfAdjointView<Eigen::SelfAdjointView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_TriangularView : std::false_type {};

    template<typename MatrixType, unsigned int UpLo>
    struct is_eigen_TriangularView<Eigen::TriangularView<MatrixType, UpLo>> : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalMatrix : std::false_type {};

    template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
    struct is_eigen_DiagonalMatrix<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
      : std::true_type {};


    template<typename T>
    struct is_eigen_DiagonalWrapper : std::false_type {};

    template<typename DiagonalVectorType>
    struct is_eigen_DiagonalWrapper<Eigen::DiagonalWrapper<DiagonalVectorType>> : std::true_type {};


    template<typename T>
    struct is_eigen_Identity : std::false_type {};

    template<typename Scalar, typename Arg>
    struct is_eigen_Identity<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
      : std::true_type {};


    template<typename T>
    struct is_eigen_MatrixWrapper : std::false_type {};

    template<typename XprType>
    struct is_eigen_MatrixWrapper<Eigen::MatrixWrapper<XprType>> : std::true_type {};


    template<typename T>
    struct is_eigen_ArrayWrapper : std::false_type {};

    template<typename XprType>
    struct is_eigen_ArrayWrapper<Eigen::ArrayWrapper<XprType>> : std::true_type {};
  }


  /**
   * \brief T is of type Eigen::SelfAdjointView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_SelfAdjointView = detail::is_eigen_SelfAdjointView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::TriangularView.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#else
  constexpr bool eigen_TriangularView = detail::is_eigen_TriangularView<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalMatrix = detail::is_eigen_DiagonalMatrix<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_DiagonalWrapper = detail::is_eigen_DiagonalWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a general eigen native type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept native_eigen_general =
#else
  constexpr bool native_eigen_general =
#endif
    (std::is_base_of_v<Eigen::EigenBase<std::decay_t<T>>, std::decay_t<T>> or native_eigen_dense<T>) and
    (not std::is_base_of_v<Eigen3Base, std::decay_t<T>>);


  /**
   * \brief T is an Eigen identity matrix (not necessarily an \ref identity_matrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#else
  constexpr bool eigen_Identity = detail::is_eigen_Identity<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_MatrixWrapper = detail::is_eigen_MatrixWrapper<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is of type Eigen::DiagonalMatrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_ArrayWrapper = detail::is_eigen_ArrayWrapper<std::decay_t<T>>::value;
#else
  constexpr bool eigen_ArrayWrapper = detail::is_eigen_ArrayWrapper<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_Eigen_traits_impl : std::false_type {};

    template<typename T>
    struct has_Eigen_traits_impl<T, std::void_t<Eigen::internal::traits<std::decay_t<T>>>> : std::true_type {};
  } // namespace detail
#endif


  /**
   * \brief Whether T has native Eigen::internal::traits.
   * \tparam T An expression
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_eigen_traits = requires { typename Eigen::internal::traits<std::decay_t<T>>; };
#else
  constexpr bool has_eigen_traits = detail::has_Eigen_traits_impl<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_eigen_evaluator_impl : std::false_type {};

    template<typename T>
    struct has_eigen_evaluator_impl<T, std::void_t<Eigen::internal::evaluator<T>>> : std::true_type {};
  }
#endif


  /**
   * \brief Whether T has native Eigen::internal::evaluator.
   * \tparam T An expression
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_eigen_evaluator = (not eigen_TriangularView<T>) and (not eigen_SelfAdjointView<T>) and
    (not eigen_DiagonalWrapper<T>) and (not eigen_DiagonalMatrix<T>) and
    requires { typename Eigen::internal::evaluator<std::decay_t<T>>; };
#else
  constexpr bool has_eigen_evaluator = (not eigen_TriangularView<T>) and (not eigen_SelfAdjointView<T>) and
    (not eigen_DiagonalWrapper<T>) and (not eigen_DiagonalMatrix<T>) and
    detail::has_eigen_evaluator_impl<std::decay_t<T>>::value;
#endif


} // namespace OpenKalman::Eigen3


namespace OpenKalman::internal
{
  using namespace OpenKalman::Eigen3;

  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

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
      (BlockRows == Eigen::Dynamic or row_dimension_of_v<U> == BlockRows) and
      (BlockRows != Eigen::Dynamic or row_dimension_of_v<U> == dynamic_size) and
      (BlockCols == Eigen::Dynamic or column_dimension_of_v<U> == BlockCols) and
      (BlockCols != Eigen::Dynamic or column_dimension_of_v<U> == dynamic_size) and
      (std::is_same_v<scalar_type_of_t<XprType>, scalar_type_of_t<U>>)> {};


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

}

#endif //OPENKALMAN_EIGEN3_MATRIX_TRAITS_HPP
