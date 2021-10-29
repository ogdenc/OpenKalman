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
 * \brief Type traits as applied to matrix classes in OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_EIGEN3_TRAITS_HPP
#define OPENKALMAN_EIGEN3_TRAITS_HPP

#include <type_traits>

// ================================================ //
//   Type traits for Eigen interface matrix types   //
// ================================================ //

namespace OpenKalman::Eigen3::internal
{
  // ----------------------------- //
  //  is_upper_triangular_storage  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_upper_triangular_storage<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::upper>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_upper_triangular_storage<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  template<typename MatrixType, unsigned int UpLo>
  struct is_upper_triangular_storage<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<(UpLo & Eigen::Upper) != 0> {};

  // ----------------------------- //
  //  is_lower_triangular_storage  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_lower_triangular_storage<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::lower>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_lower_triangular_storage<Eigen3::SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  template<typename MatrixType, unsigned int UpLo>
  struct is_lower_triangular_storage<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::bool_constant<(UpLo & Eigen::Lower) != 0> {};

} // namespace OpenKalman::Eigen3::internal


namespace OpenKalman::internal
{

  // ------------------------ //
  //  is_covariance_nestable  //
  // ------------------------ //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref covariance.
#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_constant_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))
  struct is_covariance_nestable<T>
#else
  template<typename T>
  struct is_covariance_nestable<T, std::enable_if_t<
    Eigen3::eigen_constant_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))>>
#endif
  : std::true_type {};


  // -------------------------- //
  //  is_typed_matrix_nestable  //
  // -------------------------- //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref typed_matrix.
#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_matrix<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T>
  struct is_typed_matrix_nestable<T>
#else
  template<typename T>
  struct is_typed_matrix_nestable<T, std::enable_if_t<
    Eigen3::eigen_matrix<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T>>>
#endif
  : std::true_type {};


  // ---------------- //
  //  is_zero_matrix  //
  // ---------------- //

  // The product of a zero matrix and a scalar (or vice versa) is zero.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<zero_matrix<Arg1> or zero_matrix<Arg2>> {};


  // The sum of two zero matrices is zero.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<zero_matrix<Arg1> and zero_matrix<Arg2>> {};


  // The difference between two zero or identity matrices is zero.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<(zero_matrix<Arg1> and zero_matrix<Arg2>) or
        (identity_matrix<Arg1> and identity_matrix<Arg2>)> {};


  // The negation of a zero matrix is zero.
  template<typename Scalar, typename Arg>
  struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<zero_matrix<Arg>> {};


  // The conjugate of a zero matrix is zero.
  template<typename Scalar, typename Arg>
  struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<zero_matrix<Arg>> {};


// The product of two zero matrices is zero.
  template<typename Arg1, typename Arg2>
  struct is_zero_matrix<Eigen::Product<Arg1, Arg2>> : std::bool_constant<zero_matrix<Arg1> or zero_matrix<Arg2>> {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_zero_matrix<Eigen3::ConstantMatrix<Scalar, 0, rows, columns>> : std::true_type {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_zero_matrix<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_zero_matrix<Eigen3::DiagonalMatrix<NestedMatrix>> : std::bool_constant<zero_matrix<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_zero_matrix<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<zero_matrix<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType storage_type>
  struct is_zero_matrix<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_type>>
    : std::bool_constant<zero_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_zero_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
  : std::bool_constant<Coefficients::axes_only and zero_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_zero_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
  : std::bool_constant<Coefficients::axes_only and zero_matrix<NestedMatrix>> {};


  // -------------------- //
  //  is_identity_matrix  //
  // -------------------- //

  // The conjugate of an identity matrix is also identity.
  template<typename Scalar, typename Arg>
  struct is_identity_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<identity_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_identity_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, Arg>>
    : std::bool_constant<dynamic_shape<Arg> or square_matrix<Arg>> {};


  // The product of two identity matrices is also identity.
  template<typename Arg1, typename Arg2>
  struct is_identity_matrix<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<identity_matrix<Arg1> and identity_matrix<Arg2>> {};


  template<typename Scalar>
  struct is_identity_matrix<Eigen3::ConstantMatrix<Scalar, 1, 1, 1>> : std::true_type {};


#ifdef __cpp_concepts
  template<typename NestedMatrix>
  struct is_identity_matrix<Eigen3::DiagonalMatrix<NestedMatrix>>
    : std::bool_constant<(one_by_one_matrix<NestedMatrix> and identity_matrix<NestedMatrix>) or
      (Eigen3::eigen_constant_expr<NestedMatrix> and requires { MatrixTraits<NestedMatrix>::constant == 1; })> {};
#else
  template<typename NestedMatrix>
  struct is_identity_matrix<Eigen3::DiagonalMatrix<NestedMatrix>, std::enable_if_t<
      not Eigen3::eigen_constant_expr<NestedMatrix>>>
    : std::bool_constant<one_by_one_matrix<NestedMatrix> and identity_matrix<NestedMatrix>> {};

  template<typename NestedMatrix>
  struct is_identity_matrix<Eigen3::DiagonalMatrix<NestedMatrix>, std::enable_if_t<
    Eigen3::eigen_constant_expr<NestedMatrix> and MatrixTraits<NestedMatrix>::constant == 1>> : std::true_type {};
#endif


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_identity_matrix<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<identity_matrix<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType storage_type>
  struct is_identity_matrix<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_type>>
    : std::bool_constant<identity_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_identity_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and identity_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_identity_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and identity_matrix<NestedMatrix>> {};


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

#ifdef __cpp_concepts
  template<typename Arg> requires std::derived_from<Arg, Eigen::DiagonalBase<Arg>>
  struct is_diagonal_matrix<Arg> : std::true_type {};
#else
  template<typename Arg>
  struct is_diagonal_matrix<Arg, std::enable_if_t<std::is_base_of_v<Eigen::DiagonalBase<Arg>, Arg>>>
    : std::true_type {};
#endif


  // A diagonal matrix times a scalar (or vice versa) is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_product_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> or diagonal_matrix<Arg2>> {};


  // A diagonal matrix divided by a scalar is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_quotient_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1>> {};


  // The sum of two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  // The difference between two diagonal matrices is also diagonal.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  // The negation of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  // The conjugate of a diagonal matrix is also diagonal.
  template<typename Scalar, typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<diagonal_matrix<Arg>> {};


  // The product of two diagonal matrices is also diagonal.
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>> {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_diagonal_matrix<Eigen3::ConstantMatrix<Scalar, 0, rows, columns>> : std::true_type {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_diagonal_matrix<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_diagonal_matrix<Eigen3::DiagonalMatrix<NestedMatrix>> : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_diagonal_matrix<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal> {};


  template<typename NestedMatrix, TriangleType storage_type>
  struct is_diagonal_matrix<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or storage_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
  : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
  : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  // ------------------------ //
  //  is_self_adjoint_matrix  //
  // ------------------------ //

  // The sum of two self-adjoint matrices is self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_self_adjoint_matrix<Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2>> {};


  // The difference between two self-adjoint matrices is self-adjoint.
  template<typename Scalar1, typename Scalar2, typename Arg1, typename Arg2>
  struct is_self_adjoint_matrix<
    Eigen::CwiseBinaryOp<Eigen::internal::scalar_difference_op<Scalar1, Scalar2>, Arg1, Arg2>>
    : std::bool_constant<self_adjoint_matrix<Arg1> and self_adjoint_matrix<Arg2>> {};


  // A unary operation on a self-adjoint matrix is also self-adjoint.
  template<typename UnaryOp, typename Arg>
  struct is_self_adjoint_matrix<Eigen::CwiseUnaryOp<UnaryOp, Arg>>
    : std::bool_constant<self_adjoint_matrix<Arg>> {};


  // A constant square matrix is self-adjoint.
  template<typename Scalar, typename PlainObjectType>
  struct is_self_adjoint_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType>> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::true_type {};


  template<typename Scalar, auto constant, std::size_t dim>
  struct is_self_adjoint_matrix<Eigen3::ConstantMatrix<Scalar, constant, dim, dim>> : std::true_type {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_self_adjoint_matrix<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_self_adjoint_matrix<Eigen3::DiagonalMatrix<NestedMatrix>> : std::true_type {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_self_adjoint_matrix<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_self_adjoint_matrix<Eigen3::TriangularMatrix<NestedMatrix, TriangleType::diagonal>> : std::true_type {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_self_adjoint_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and self_adjoint_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_self_adjoint_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and self_adjoint_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_lower_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_lower_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<lower_triangular_matrix<Arg>> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_lower_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<lower_triangular_matrix<MatrixType> or (Mode & Eigen::Lower) != 0> {};


  template<typename Scalar, auto constant, std::size_t dim>
  struct is_lower_triangular_matrix<Eigen3::ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<constant == 0 or dim == 1> {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_lower_triangular_matrix<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_lower_triangular_matrix<Eigen3::DiagonalMatrix<NestedMatrix>> : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_lower_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<lower_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::lower or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  template<typename Scalar, typename Arg>
  struct is_upper_triangular_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_conjugate_op<Scalar>, Arg>>
    : std::bool_constant<upper_triangular_matrix<Arg>> {};


  template<typename MatrixType, unsigned int Mode>
  struct is_upper_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<upper_triangular_matrix<MatrixType> or (Mode & Eigen::Upper) != 0> {};


  template<typename Scalar, auto constant, std::size_t dim>
  struct is_upper_triangular_matrix<Eigen3::ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<constant == 0 or dim == 1> {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_upper_triangular_matrix<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct is_upper_triangular_matrix<Eigen3::DiagonalMatrix<NestedMatrix>> : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_upper_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<upper_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::upper or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  // ------------------ //
  //  has_dynamic_rows  //
  // ------------------ //

#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires (T::RowsAtCompileTime == Eigen::Dynamic) //< Eigen::Dynamic is -1
  struct has_dynamic_rows<T>
#else
  template<typename T>
  struct has_dynamic_rows<T, std::enable_if_t<Eigen3::eigen_native<T> and (T::RowsAtCompileTime == Eigen::Dynamic)>>
#endif
  : std::true_type {};


  template<typename Scalar, std::size_t columns>
  struct has_dynamic_rows<Eigen3::ZeroMatrix<Scalar, 0, columns>> : std::true_type {};


  template<typename Scalar, auto constant, std::size_t columns>
  struct has_dynamic_rows<Eigen3::ConstantMatrix<Scalar, constant, 0, columns>> : std::true_type {};


  template<typename NestedMatrix>
  struct has_dynamic_rows<Eigen3::DiagonalMatrix<NestedMatrix>> : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct has_dynamic_rows<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct has_dynamic_rows<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct has_dynamic_rows<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct has_dynamic_rows<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  // --------------------- //
  //  has_dynamic_columns  //
  // --------------------- //

#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires (T::ColsAtCompileTime == Eigen::Dynamic) //< Eigen::Dynamic is -1
  struct has_dynamic_columns<T>
#else
  template<typename T>
  struct has_dynamic_columns<T, std::enable_if_t<
      Eigen3::eigen_native<T> and (T::ColsAtCompileTime == Eigen::Dynamic)>>
#endif
  : std::true_type {};


  template<typename Scalar, std::size_t rows>
  struct has_dynamic_columns<Eigen3::ZeroMatrix<Scalar, rows, 0>> : std::true_type {};


  template<typename Scalar, auto constant, std::size_t rows>
  struct has_dynamic_columns<Eigen3::ConstantMatrix<Scalar, constant, rows, 0>> : std::true_type {};


  template<typename NestedMatrix>
  struct has_dynamic_columns<Eigen3::DiagonalMatrix<NestedMatrix>> : std::bool_constant<dynamic_rows<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct has_dynamic_columns<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<dynamic_columns<NestedMatrix>> {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct has_dynamic_columns<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<dynamic_columns<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct has_dynamic_columns<Eigen3::ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<dynamic_columns<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct has_dynamic_columns<Eigen3::FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<dynamic_columns<NestedMatrix>> {};


  // ------------------- //
  //  is_self_contained  //
  // ------------------- //

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


  template<typename XprType, int BlockRows, int BlockCols, bool InnerPanel>
  struct is_self_contained<Eigen::Block<XprType, BlockRows, BlockCols, InnerPanel>>
    : std::bool_constant<detail::stores<XprType>> {};


  template<typename BinaryOp, typename LhsType, typename RhsType>
  struct is_self_contained<Eigen::CwiseBinaryOp<BinaryOp, LhsType, RhsType>>
    : std::bool_constant<detail::stores<LhsType> and detail::stores<RhsType>> {};


  template<typename Scalar, typename PlainObjectType>
  struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<Scalar>, PlainObjectType>>
    : std::true_type {};


  template<typename Scalar, typename PlainObjectType>
  struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::true_type {};


  template<typename Scalar, typename PacketType, typename PlainObjectType>
  struct is_self_contained<Eigen::CwiseNullaryOp<Eigen::internal::linspaced_op<Scalar, PacketType>, PlainObjectType>>
    : std::true_type {};


  template<typename TernaryOp, typename Arg1, typename Arg2, typename Arg3>
  struct is_self_contained<Eigen::CwiseTernaryOp<TernaryOp, Arg1, Arg2, Arg3>>
    : std::bool_constant<detail::stores<Arg1> and detail::stores<Arg2> and detail::stores<Arg3>> {};


  template<typename UnaryOp, typename XprType>
  struct is_self_contained<Eigen::CwiseUnaryOp<UnaryOp, XprType>>
    : std::bool_constant<detail::stores<XprType>> {};


  template<typename MatrixType, int DiagIndex>
  struct is_self_contained<Eigen::Diagonal<MatrixType, DiagIndex>>
    : std::bool_constant<detail::stores<MatrixType>> {};


  template<typename Scalar, int SizeAtCompileTime, int MaxSizeAtCompileTime>
  struct is_self_contained<Eigen::DiagonalMatrix<Scalar, SizeAtCompileTime, MaxSizeAtCompileTime>>
    : std::true_type {};


  template<typename DiagVectorType>
  struct is_self_contained<Eigen::DiagonalWrapper<DiagVectorType>>
    : std::false_type {};


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
    : std::false_type {};


  template<typename Decomposition, typename RhsType>
  struct is_self_contained<Eigen::Solve<Decomposition, RhsType>>
    : std::bool_constant<detail::stores<Decomposition> and detail::stores<RhsType>> {};


  template<typename MatrixType>
  struct is_self_contained<Eigen::Transpose<MatrixType>>
    : std::bool_constant<detail::stores<MatrixType>> {};


  template<typename MatrixType, unsigned int UpLo>
  struct is_self_contained<Eigen::TriangularView<MatrixType, UpLo>>
    : std::false_type {};


  template<typename VectorType, int Size>
  struct is_self_contained<Eigen::VectorBlock<VectorType, Size>>
    : std::bool_constant<detail::stores<VectorType>> {};

  // New OpenKalman types //

  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
  struct is_self_contained<Eigen3::ConstantMatrix<Scalar, constant, rows, columns>> : std::true_type {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_self_contained<Eigen3::ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T>
  struct is_self_contained<T> : std::bool_constant<self_contained<nested_matrix_t<T>>> {};
#else
  template<typename T>
  struct is_self_contained<T, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T>>> : std::bool_constant<self_contained<nested_matrix_t<T>>> {};
#endif


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

#ifdef __cpp_concepts
  template<Eigen3::eigen_native T>
  struct is_element_gettable<T, 2>
#else
  template<typename T>
  struct is_element_gettable<T, 2, std::enable_if_t<Eigen3::eigen_native<T>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires column_vector<T>
  struct is_element_gettable<T, 1>
#else
  template<typename T>
  struct is_element_gettable<T, 1, std::enable_if_t<Eigen3::eigen_native<T> and column_vector<T>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_constant_expr T, std::size_t N> requires (N == 2) or (N == 1 and column_vector<T>)
  struct is_element_gettable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_constant_expr<T> and
    ((N == 2) or (N == 1 and column_vector<T>))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr T, std::size_t N> requires (N == 2) or (N == 1 and column_vector<T>)
  struct is_element_gettable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T> and
    ((N == 2) or (N == 1 and column_vector<T>))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr T, std::size_t N> requires (N <= 2) and
    (element_gettable<nested_matrix_t<T>, 2> or
      element_gettable<nested_matrix_t<T>, 1>)
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_diagonal_expr<T> and (N <= 2)>>
    : std::bool_constant<element_gettable<nested_matrix_t<T>, 2> or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr T, std::size_t N> requires
    (element_gettable<nested_matrix_t<T>, 2> and
      (N == 2 or MatrixTraits<T>::storage_triangle == TriangleType::diagonal)) or
    element_gettable<nested_matrix_t<T>, 1>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
    : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::storage_triangle == TriangleType::diagonal)) or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr T, std::size_t N> requires
    (element_gettable<nested_matrix_t<T>, 2> and
      (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
    element_gettable<nested_matrix_t<T>, 1>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
    : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::to_euclidean_expr T, std::size_t N> requires
    element_gettable<nested_matrix_t<T>, N>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::to_euclidean_expr<T>>>
    : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::from_euclidean_expr T, std::size_t N> requires
    element_gettable<nested_matrix_t<T>, N>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T>>>
    : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


  // --------------------- //
  //  is_element_settable  //
  // --------------------- //

#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires (not std::is_const_v<std::remove_reference_t<T>>) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
  struct is_element_settable<T, 2>
#else
  template<typename T>
  struct is_element_settable<T, 2, std::enable_if_t<
    Eigen3::eigen_native<T> and (not std::is_const_v<std::remove_reference_t<T>>) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_native T> requires column_vector<T> and
    (not std::is_const_v<std::remove_reference_t<T>>) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))
  struct is_element_settable<T, 1>
#else
  template<typename T>
  struct is_element_settable<T, 1, std::enable_if_t<Eigen3::eigen_native<T> and column_vector<T> and
    (not std::is_const_v<std::remove_reference_t<T>>) and
    (static_cast<bool>(std::decay_t<T>::Flags & Eigen::LvalueBit))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_constant_expr T, std::size_t N>
  struct is_element_settable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_constant_expr<T>>>
#endif
    : std::false_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr T, std::size_t N>
  struct is_element_settable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
    : std::false_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr T, std::size_t N> requires (N <= 2) and
    (element_settable<nested_matrix_t<T>, 2> or
      element_settable<nested_matrix_t<T>, 1>)
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_diagonal_expr<T> and (N <= 2)>>
    : std::bool_constant<element_settable<nested_matrix_t<T>, 2> or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr T, std::size_t N> requires
    (element_settable<nested_matrix_t<T>, 2> and
      (N == 2 or MatrixTraits<T>::storage_triangle == TriangleType::diagonal)) or
    element_settable<nested_matrix_t<T>, 1>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
    : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::storage_triangle == TriangleType::diagonal)) or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr T, std::size_t N> requires
    (element_settable<nested_matrix_t<T>, 2> and
      (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
    element_settable<nested_matrix_t<T>, 1>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
    : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or MatrixTraits<T>::triangle_type == TriangleType::diagonal)) or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::to_euclidean_expr T, std::size_t N> requires
    MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::to_euclidean_expr<T> and
    MatrixTraits<T>::RowCoefficients::axes_only>>
    : is_element_settable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::from_euclidean_expr T, std::size_t N> requires
    (MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>) or
    (Eigen3::to_euclidean_expr<nested_matrix_t<T>> and
      element_settable<nested_matrix_t<nested_matrix_t<T>>, N>)
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T> and
    (not Eigen3::to_euclidean_expr<nested_matrix_t<T>>)>>
    : std::bool_constant<MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>> {};

  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::from_euclidean_expr<T> and
    Eigen3::to_euclidean_expr<nested_matrix_t<T>>>>
    : std::bool_constant<element_settable<nested_matrix_t<nested_matrix_t<T>>, N>> {};
#endif


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
      (BlockRows == Eigen::Dynamic or MatrixTraits<U>::rows == BlockRows) and
      (BlockRows != Eigen::Dynamic or MatrixTraits<U>::rows == 0) and
      (BlockCols == Eigen::Dynamic or MatrixTraits<U>::columns == BlockCols) and
      (BlockCols != Eigen::Dynamic or MatrixTraits<U>::columns == 0) and
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

  // New OpenKalman types //

  // Note: Eigen3::ZeroMatrix is already covered because it derives from Eigen::internal::no_assignment_operator.
  template<typename Scalar, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<Eigen3::ZeroMatrix<Scalar, rows, columns>, U>
    : std::false_type {};


  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<Eigen3::ConstantMatrix<Scalar, constant, rows, columns>, U>
    : std::false_type {};


  template<typename N1, typename N2>
  struct is_modifiable_native<Eigen3::DiagonalMatrix<N1>, Eigen3::DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<Eigen3::DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType storage_triangle, typename U> requires
    //(not self_adjoint_matrix<U>) or
    //(storage_triangle == TriangleType::diagonal and not diagonal_matrix<U>) or
    (Eigen3::eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix>) or
    (not Eigen3::eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>, U> : std::false_type {};
#else
  template<typename N1, TriangleType t1, typename N2, TriangleType t2>
  struct is_modifiable_native<Eigen3::SelfAdjointMatrix<N1, t1>, Eigen3::SelfAdjointMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, TriangleType t, typename U>
  struct is_modifiable_native<Eigen3::SelfAdjointMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*self_adjoint_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
      */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type, typename U> requires
    //(not triangular_matrix<U>) or
    //(triangle_type == TriangleType::diagonal and not diagonal_matrix<U>) or
    (Eigen3::eigen_triangular_expr<U> and not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix>) or
    (not Eigen3::eigen_triangular_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<Eigen3::TriangularMatrix<NestedMatrix, triangle_type>, U> : std::false_type {};
#else
  template<typename N1, TriangleType t1, typename N2, TriangleType t2>
  struct is_modifiable_native<Eigen3::TriangularMatrix<N1, t1>, Eigen3::TriangularMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, TriangleType t, typename U>
  struct is_modifiable_native<Eigen3::TriangularMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*triangular_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
      */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (Eigen3::euclidean_expr<U> and (Eigen3::to_euclidean_expr<U> or
      not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
    (not Eigen3::euclidean_expr<U> and not modifiable<NestedMatrix, Eigen3::ToEuclideanExpr<C, std::decay_t<U>>>)
  struct is_modifiable_native<Eigen3::FromEuclideanExpr<C, NestedMatrix>, U>
    : std::false_type {};
#else
  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<Eigen3::FromEuclideanExpr<C1, N1>, Eigen3::FromEuclideanExpr<C2, N2>>
    : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<Eigen3::FromEuclideanExpr<C1, N1>, Eigen3::ToEuclideanExpr<C2, N2>>
    : std::false_type {};

  template<typename C, typename NestedMatrix, typename U>
  struct is_modifiable_native<Eigen3::FromEuclideanExpr<C, NestedMatrix>, U>
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::dimensions == MatrixTraits<U>::rows and
      MatrixTraits<NestedMatrix>::columns == MatrixTraits<U>::columns and
      std::is_same_v<typename MatrixTraits<NestedMatrix>::Scalar, typename MatrixTraits<U>::Scalar>*/> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (Eigen3::euclidean_expr<U> and (Eigen3::from_euclidean_expr<U> or
      not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
    (not Eigen3::euclidean_expr<U> and not modifiable<NestedMatrix, Eigen3::FromEuclideanExpr<C, std::decay_t<U>>>)
  struct is_modifiable_native<Eigen3::ToEuclideanExpr<C, NestedMatrix>, U>
  : std::false_type {};
#else
  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<Eigen3::ToEuclideanExpr<C1, N1>, Eigen3::ToEuclideanExpr<C2, N2>>
    : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

  template<typename C1, typename N1, typename C2, typename N2>
  struct is_modifiable_native<Eigen3::ToEuclideanExpr<C1, N1>, Eigen3::FromEuclideanExpr<C2, N2>>
    : std::false_type {};

  template<typename C, typename NestedMatrix, typename U>
  struct is_modifiable_native<Eigen3::ToEuclideanExpr<C, NestedMatrix>, U, std::void_t<Eigen3::FromEuclideanExpr<C, std::decay_t<U>>>>
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::euclidean_dimensions == MatrixTraits<U>::rows and
      MatrixTraits<NestedMatrix>::columns == MatrixTraits<U>::columns and
      std::is_same_v<typename MatrixTraits<NestedMatrix>::Scalar, typename MatrixTraits<U>::Scalar>*/> {};
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
