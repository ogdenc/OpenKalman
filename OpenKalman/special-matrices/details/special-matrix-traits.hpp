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
 * \brief Type traits as applied to special matrix classes in OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP
#define OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP

#include <type_traits>

// ================================================ //
//   Type traits for Eigen interface matrix types   //
// ================================================ //

namespace OpenKalman
{

  namespace internal
  {

    // ---------------------- //
    //  is_modifiable_native  //
    // ---------------------- //

    template<typename N1, typename N2>
    struct is_modifiable_native<DiagonalMatrix<N1>, DiagonalMatrix<N2>>
      : std::bool_constant<modifiable<N1, N2>> {};


    template<typename NestedMatrix, typename U>
    struct is_modifiable_native<DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
    template<typename NestedMatrix, HermitianAdapterType storage_triangle, typename U> requires
      //(not hermitian_matrix<U>) or
      (eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, nested_matrix_of_t<U>>) or
      (not eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, U>)
    struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, storage_triangle>, U> : std::false_type {};
#else
    template<typename N1, HermitianAdapterType t1, typename N2, HermitianAdapterType t2>
    struct is_modifiable_native<SelfAdjointMatrix<N1, t1>, SelfAdjointMatrix<N2, t2>>
      : std::bool_constant<modifiable<N1, N2>> {};

    template<typename NestedMatrix, HermitianAdapterType t, typename U>
    struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, t>, U>
      : std::bool_constant</*hermitian_matrix<U> and */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
    template<typename NestedMatrix, TriangleType triangle_type, typename U> requires
      //(not triangular_matrix<U>) or
      //(triangle_type == TriangleType::diagonal and not diagonal_matrix<U>) or
      (eigen_triangular_expr<U> and not modifiable<NestedMatrix, nested_matrix_of_t<U>>) or
      (not eigen_triangular_expr<U> and not modifiable<NestedMatrix, U>)
    struct is_modifiable_native<TriangularMatrix<NestedMatrix, triangle_type>, U> : std::false_type {};
#else
    template<typename N1, TriangleType t1, typename N2, TriangleType t2>
    struct is_modifiable_native<TriangularMatrix<N1, t1>, TriangularMatrix<N2, t2>>
      : std::bool_constant<modifiable<N1, N2>> {};

    template<typename NestedMatrix, TriangleType t, typename U>
    struct is_modifiable_native<TriangularMatrix<NestedMatrix, t>, U>
      : std::bool_constant</*triangular_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
        */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
    template<typename C, typename NestedMatrix, typename U> requires
      (euclidean_expr<U> and (to_euclidean_expr<U> or
        not modifiable<NestedMatrix, nested_matrix_of_t<U>> or
        not equivalent_to<C, row_index_descriptor_of_t<U>>)) or
      (not euclidean_expr<U> and not modifiable<NestedMatrix, ToEuclideanExpr<C, std::decay_t<U>>>)
    struct is_modifiable_native<FromEuclideanExpr<C, NestedMatrix>, U>
      : std::false_type {};
#else
    template<typename C1, typename N1, typename C2, typename N2>
    struct is_modifiable_native<FromEuclideanExpr<C1, N1>, FromEuclideanExpr<C2, N2>>
      : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

    template<typename C1, typename N1, typename C2, typename N2>
    struct is_modifiable_native<FromEuclideanExpr<C1, N1>, ToEuclideanExpr<C2, N2>>
      : std::false_type {};

    template<typename C, typename NestedMatrix, typename U>
    struct is_modifiable_native<FromEuclideanExpr<C, NestedMatrix>, U>
      : std::bool_constant<modifiable<NestedMatrix, dense_writable_matrix_t<NestedMatrix, scalar_type_of_t<NestedMatrix>>>/* and dimension_size_of_v<C> == row_dimension_of_v<U> and
        column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
        std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif


#ifdef __cpp_concepts
    template<typename C, typename NestedMatrix, typename U> requires
      (euclidean_expr<U> and (from_euclidean_expr<U> or
        not modifiable<NestedMatrix, nested_matrix_of_t<U>> or
        not equivalent_to<C, row_index_descriptor_of_t<U>>)) or
      (not euclidean_expr<U> and not modifiable<NestedMatrix, FromEuclideanExpr<C, std::decay_t<U>>>)
    struct is_modifiable_native<ToEuclideanExpr<C, NestedMatrix>, U>
    : std::false_type {};
#else
    template<typename C1, typename N1, typename C2, typename N2>
    struct is_modifiable_native<ToEuclideanExpr<C1, N1>, ToEuclideanExpr<C2, N2>>
      : std::bool_constant<modifiable<N1, N2> and equivalent_to<C1, C2>> {};

    template<typename C1, typename N1, typename C2, typename N2>
    struct is_modifiable_native<ToEuclideanExpr<C1, N1>, FromEuclideanExpr<C2, N2>>
      : std::false_type {};

    template<typename C, typename NestedMatrix, typename U>
    struct is_modifiable_native<ToEuclideanExpr<C, NestedMatrix>, U, std::void_t<FromEuclideanExpr<C, std::decay_t<U>>>>
      : std::bool_constant<modifiable<NestedMatrix, dense_writable_matrix_t<NestedMatrix, scalar_type_of_t<NestedMatrix>>>/* and euclidean_dimension_size_of_v<C> == row_dimension_of_v<U> and
        column_dimension_of_v<NestedMatrix> == column_dimension_of_v<U> and
        std::is_same_v<scalar_type_of_t<NestedMatrix>, scalar_type_of_t<U>>*/> {};
#endif

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_SPECIAL_MATRIX_TRAITS_HPP
