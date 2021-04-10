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
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))
  struct is_covariance_nestable<T>
#else
  template<typename T>
  struct is_covariance_nestable<T, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    (Eigen3::eigen_native<T> and (triangular_matrix<T> or self_adjoint_matrix<T>))>>
#endif
  : std::true_type {};


  // ------------------------ //
  //  is_typed_matrix_nestable  //
  // ------------------------ //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref typed_matrix.
#ifdef __cpp_concepts
  template<typename T> requires
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T> or
    Eigen3::eigen_native<T>
  struct is_typed_matrix_nestable<T>
#else
  template<typename T>
  struct is_typed_matrix_nestable<T, std::enable_if_t<
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::to_euclidean_expr<T> or
    Eigen3::from_euclidean_expr<T> or
    Eigen3::eigen_native<T>>>
#endif
  : std::true_type {};


  // ------------------------------------------ //
  //  is_element_gettable, is_element_settable  //
  // ------------------------------------------ //

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


  // ---------------- //
  //  is_zero_matrix  //
  // ---------------- //

#ifdef __cpp_concepts
  template<typename T> requires (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T>) and zero_matrix<nested_matrix_t<T>>
  struct is_zero_matrix<T>
    : std::true_type {};
#else
  template<typename T>
  struct is_zero_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T> or
    Eigen3::eigen_diagonal_expr<T>>>
    : is_zero_matrix<nested_matrix_t<T>> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr T>
  struct is_zero_matrix<T>
#else
  template<typename T>
  struct is_zero_matrix<T, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
    : std::true_type {};


  // The product of two zero matrices is zero.
  template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
  requires zero_matrix<Arg1> or zero_matrix<Arg2>
  struct is_zero_matrix<Eigen::Product<Arg1, Arg2>>
#else
  struct is_zero_matrix<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<zero_matrix<Arg1> or zero_matrix<Arg2>>>
#endif
    : std::true_type {};


  // The product of a zero matrix and a scalar (or vice versa) is zero.
  template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
  requires zero_matrix<Arg1> or zero_matrix<Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<zero_matrix<Arg1> or zero_matrix<Arg2>>>
#endif
    : std::true_type {};


  // The sum of two zero matrices is zero.
#ifdef __cpp_concepts
  template<zero_matrix Arg1, zero_matrix Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<zero_matrix<Arg1> and zero_matrix<Arg2>>>
#endif
    : std::true_type {};


  // The difference between two zero or identity matrices is zero.
  template<typename Arg1, typename Arg2>
#ifdef __cpp_concepts
  requires (zero_matrix<Arg1> and zero_matrix<Arg2>) or (identity_matrix<Arg1> and identity_matrix<Arg2>)
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  struct is_zero_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<(zero_matrix<Arg1> and zero_matrix<Arg2>) or (identity_matrix<Arg1> and identity_matrix<Arg2>)>>
#endif
    : std::true_type {};


  // The negation of a zero matrix is zero.
#ifdef __cpp_concepts
  template<zero_matrix Arg>
  struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>>
#else
  template<typename Arg>
  struct is_zero_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<zero_matrix<Arg>>>
#endif
    : std::true_type {};


  // -------------------- //
  //  is_identity_matrix  //
  // -------------------- //

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr T> requires one_by_one_matrix<T> and identity_matrix<nested_matrix_t<T>>
  struct is_identity_matrix<T>
#else
  template<typename T>
  struct is_identity_matrix<T, std::enable_if_t<
    Eigen3::eigen_diagonal_expr<T> and one_by_one_matrix<T> and identity_matrix<nested_matrix_t<T>>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<typename T> requires (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>) and
    identity_matrix<nested_matrix_t<T>>
  struct is_identity_matrix<T>
#else
  template<typename T>
  struct is_identity_matrix<T, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<T> or Eigen3::eigen_triangular_expr<T>) and identity_matrix<nested_matrix_t<T>>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::euclidean_expr T> requires
  MatrixTraits<T>::Coefficients::axes_only and identity_matrix<nested_matrix_t<T>>
  struct is_identity_matrix<T>
#else
  template<typename T>
  struct is_identity_matrix<T, std::enable_if_t<Eigen3::euclidean_expr<T> and
    MatrixTraits<T>::Coefficients::axes_only and identity_matrix<nested_matrix_t<T>>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<typename Arg> requires (Arg::RowsAtCompileTime == Arg::ColsAtCompileTime)
  struct is_identity_matrix<Eigen3::IdentityMatrix<Arg>> : std::true_type {};
#else
  template<typename Arg>
  struct is_identity_matrix<Eigen3::IdentityMatrix<Arg>>
    : std::bool_constant<Arg::RowsAtCompileTime == Arg::ColsAtCompileTime> {};
#endif


  // The product of two identity matrices is also identity.
#ifdef __cpp_concepts
  template<identity_matrix Arg1, identity_matrix Arg2>
  struct is_identity_matrix<Eigen::Product<Arg1, Arg2>> : std::true_type {};
#else
  template<typename Arg1, typename Arg2>
  struct is_identity_matrix<Eigen::Product<Arg1, Arg2>>
    : std::bool_constant<identity_matrix<Arg1> and identity_matrix<Arg2>> {};
#endif


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

#ifdef __cpp_concepts
  template<Eigen3::eigen_diagonal_expr T>
  struct is_diagonal_matrix<T>
#else
  template<typename T>
  struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_diagonal_expr<T>>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<Eigen3::eigen_self_adjoint_expr T> requires diagonal_matrix<nested_matrix_t<T>> or
    (MatrixTraits<T>::storage_triangle == TriangleType::diagonal)
  struct is_diagonal_matrix<T> : std::true_type {};
#else
  template<typename T>
  struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_self_adjoint_expr<T>>>
    : std::bool_constant<diagonal_matrix<nested_matrix_t<T>> or
        MatrixTraits<T>::storage_triangle == TriangleType::diagonal> {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_triangular_expr T> requires diagonal_matrix<nested_matrix_t<T>> or
    (MatrixTraits<T>::triangle_type == TriangleType::diagonal)
  struct is_diagonal_matrix<T> : std::true_type {};
#else
  template<typename T>
  struct is_diagonal_matrix<T, std::enable_if_t<Eigen3::eigen_triangular_expr<T>>>
    : std::bool_constant<diagonal_matrix<nested_matrix_t<T>> or
        MatrixTraits<T>::triangle_type == TriangleType::diagonal> {};
#endif


#ifdef __cpp_concepts
  template<typename Arg> requires std::derived_from<Arg, Eigen::DiagonalBase<Arg>>
  struct is_diagonal_matrix<Arg> : std::true_type {};
#else
  template<typename Arg>
  struct is_diagonal_matrix<Arg, std::enable_if_t<std::is_base_of_v<Eigen::DiagonalBase<Arg>, Arg>>>
    : std::true_type {};
#endif


  // The product of two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
  template<diagonal_matrix Arg1, diagonal_matrix Arg2>
  struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::Product<Arg1, Arg2>,
    std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
    : std::true_type {};


  // A diagonal matrix times a scalar (or vice versa) is also diagonal.
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires diagonal_matrix<Arg1> or diagonal_matrix<Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_product_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<diagonal_matrix<Arg1> or diagonal_matrix<Arg2>>>
#endif
    : std::true_type {};


  // A diagonal matrix divided by a scalar is also diagonal.
#ifdef __cpp_concepts
  template<diagonal_matrix Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_quotient_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<diagonal_matrix<Arg1>>>
#endif
    : std::true_type {};


  // The sum of two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
  template<diagonal_matrix Arg1, diagonal_matrix Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_sum_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
    : std::true_type {};


  // The difference between two diagonal matrices is also diagonal.
#ifdef __cpp_concepts
  template<diagonal_matrix Arg1, diagonal_matrix Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>>
#else
  template<typename Arg1, typename Arg2>
  struct is_diagonal_matrix<Eigen::CwiseBinaryOp<
    Eigen::internal::scalar_difference_op<typename Arg1::Scalar, typename Arg2::Scalar>, Arg1, Arg2>,
    std::enable_if_t<diagonal_matrix<Arg1> and diagonal_matrix<Arg2>>>
#endif
    : std::true_type {};


  // The negation of an identity matrix is diagonal.
#ifdef __cpp_concepts
  template<diagonal_matrix Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>>
#else
  template<typename Arg>
  struct is_diagonal_matrix<Eigen::CwiseUnaryOp<Eigen::internal::scalar_opposite_op<typename Arg::Scalar>, Arg>,
    std::enable_if_t<diagonal_matrix<Arg>>>
#endif
    : std::true_type {};


  // ------------------------ //
  //  is_self_adjoint_matrix  //
  // ------------------------ //

  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_self_adjoint_matrix<Eigen3::SelfAdjointMatrix<NestedMatrix, storage_triangle>> : std::true_type {};

  template<typename Scalar, typename PlainObjectType>
  struct is_self_adjoint_matrix<Eigen::CwiseNullaryOp<Eigen::internal::scalar_constant_op<Scalar>, PlainObjectType>>
    : std::bool_constant<square_matrix<PlainObjectType>> {};

  template<typename MatrixType, unsigned int UpLo>
  struct is_self_adjoint_matrix<Eigen::SelfAdjointView<MatrixType, UpLo>>
    : std::true_type {};


  // ---------------------------- //
  //  is_lower_triangular_matrix  //
  // ---------------------------- //

  template<typename NestedMatrix>
  struct is_lower_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, TriangleType::lower>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_lower_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  template<typename MatrixType, unsigned int Mode>
  struct is_lower_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<(Mode & Eigen::Lower) != 0> {};


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  template<typename NestedMatrix>
  struct is_upper_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, TriangleType::upper>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_upper_triangular_matrix<Eigen3::TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  template<typename MatrixType, unsigned int Mode>
  struct is_upper_triangular_matrix<Eigen::TriangularView<MatrixType, Mode>>
    : std::bool_constant<(Mode & Eigen::Upper) != 0> {};


  // ------------------- //
  //  is_self_contained  //
  // ------------------- //

#ifdef __cpp_concepts
  template<typename T> requires
    (Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::to_euclidean_expr<T> or
      Eigen3::from_euclidean_expr<T>) and
    self_contained<nested_matrix_t<T>>
  struct is_self_contained<T> : std::true_type {};
#else
  template<typename T>
  struct is_self_contained<T, std::enable_if_t<
    (Eigen3::eigen_self_adjoint_expr<T> or
      Eigen3::eigen_triangular_expr<T> or
      Eigen3::eigen_diagonal_expr<T> or
      Eigen3::to_euclidean_expr<T> or
      Eigen3::from_euclidean_expr<T>) and
    self_contained<nested_matrix_t<T>>>> : std::true_type {};
#endif


#ifdef __cpp_concepts
  template<Eigen3::eigen_zero_expr T>
  struct is_self_contained<T>
#else
  template<typename T>
  struct is_self_contained<T, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
    : std::true_type {};


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

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
  template<Eigen3::eigen_zero_expr T, std::size_t N> requires (N == 2) or (N == 1 and column_vector<T>)
  struct is_element_gettable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T> and
    ((N == 2) or (N == 1 and column_vector<T>))>>
#endif
    : std::true_type {};


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
  template<Eigen3::eigen_zero_expr T, std::size_t N>
  struct is_element_settable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<Eigen3::eigen_zero_expr<T>>>
#endif
    : std::false_type {};


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
  template<typename NestedMatrix, typename U> requires (not diagonal_matrix<U>) or
    (Eigen3::eigen_diagonal_expr<U> and not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix>) or
    (not Eigen3::eigen_diagonal_expr<U> and
      not requires { modifiable<NestedMatrix, decltype(std::declval<U>().diagonal())>; })
  struct is_modifiable_native<Eigen3::DiagonalMatrix<NestedMatrix>, U> : std::false_type {};
#else
  template<typename N1, typename N2>
  struct is_modifiable_native<Eigen3::DiagonalMatrix<N1>, Eigen3::DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<Eigen3::DiagonalMatrix<NestedMatrix>, U>
    : std::bool_constant</*diagonal_matrix<U> and */modifiable<NestedMatrix, decltype(std::declval<U>().diagonal())>> {};
#endif


  // Note: Eigen3::ZeroMatrix is already covered because it derives from Eigen::internal::no_assignment_operator.
#ifdef __cpp_concepts
  template<typename Scalar, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<Eigen3::ZeroMatrix<Scalar, rows, columns>, U>
    : std::false_type {};
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
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::size == MatrixTraits<U>::dimension and
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
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::euclidean_dimension == MatrixTraits<U>::dimension and
      MatrixTraits<NestedMatrix>::columns == MatrixTraits<U>::columns and
      std::is_same_v<typename MatrixTraits<NestedMatrix>::Scalar, typename MatrixTraits<U>::Scalar>*/> {};
#endif

} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_TRAITS_HPP
