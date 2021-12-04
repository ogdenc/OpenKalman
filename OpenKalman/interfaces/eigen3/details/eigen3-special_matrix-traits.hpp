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

#ifndef OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP
#define OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP

#include <type_traits>

// ================================================ //
//   Type traits for Eigen interface matrix types   //
// ================================================ //

namespace OpenKalman
{
  using namespace OpenKalman::Eigen3;
  using namespace OpenKalman::internal;


  // ----------------------------- //
  //  is_upper_self_adjoint_matrix  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::upper>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  // ----------------------------- //
  //  is_lower_self_adjoint_matrix  //
  // ----------------------------- //

  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::lower>>
    : std::true_type {};

  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::true_type {};

  // ------------------------ //
  //  is_covariance_nestable  //
  // ------------------------ //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref covariance.
#ifdef __cpp_concepts
  template<typename T> requires
    eigen_constant_expr<T> or
    eigen_zero_expr<T> or
    eigen_diagonal_expr<T> or
    eigen_self_adjoint_expr<T> or
    eigen_triangular_expr<T>
  struct is_covariance_nestable<T>
#else
  template<typename T>
  struct is_covariance_nestable<T, std::enable_if_t<
    Eigen3::eigen_constant_expr<T> or
    Eigen3::eigen_zero_expr<T> or
    Eigen3::eigen_diagonal_expr<T> or
    Eigen3::eigen_self_adjoint_expr<T> or
    Eigen3::eigen_triangular_expr<T>>>
#endif
  : std::true_type {};


  // -------------------------- //
  //  is_typed_matrix_nestable  //
  // -------------------------- //

  /// \internal Defines a type in Eigen3 that is nestable within a \ref typed_matrix.
#ifdef __cpp_concepts
  template<typename T>
  requires eigen_zero_expr<T> or eigen_constant_expr<T>
  struct is_typed_matrix_nestable<T>
#else
  template<typename T>
  struct is_typed_matrix_nestable<T, std::enable_if_t<eigen_zero_expr<T> or eigen_constant_expr<T>>>
#endif
  : std::true_type {};


  // ---------------------- //
  //  constant_coefficient  //
  // ---------------------- //

  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
  struct constant_coefficient<ConstantMatrix<Scalar, constant, rows, columns>>
    : constant_coefficient_type<constant> {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct constant_coefficient<ZeroMatrix<Scalar, rows, columns>>
    : constant_coefficient_type<short {0}, Scalar> {};


#ifdef __cpp_concepts
  template<zero_matrix NestedMatrix>
  struct constant_coefficient<DiagonalMatrix<NestedMatrix>>
#else
  template<typename NestedMatrix>
  struct constant_coefficient<DiagonalMatrix<NestedMatrix>, std::enable_if_t<zero_matrix<NestedMatrix>>
#endif
    : constant_coefficient_type<short {0}, typename MatrixTraits<NestedMatrix>::Scalar> {};


#ifdef __cpp_concepts
  template<zero_matrix NestedMatrix, TriangleType triangle_type>
  struct constant_coefficient<TriangularMatrix<NestedMatrix, triangle_type>>
#else
  template<typename NestedMatrix, TriangleType triangle_type>
  struct constant_coefficient<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<zero_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<short {0}, typename MatrixTraits<NestedMatrix>::Scalar> {};


#ifdef __cpp_concepts
  template<constant_matrix NestedMatrix, TriangleType storage_type>
  struct constant_coefficient<SelfAdjointMatrix<NestedMatrix, storage_type>>
#else
  template<typename NestedMatrix, TriangleType storage_type>
  struct constant_coefficient<SelfAdjointMatrix<NestedMatrix, storage_type>,
    std::enable_if_t<constant_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<typename Coefficients, constant_matrix NestedMatrix> requires Coefficients::axes_only
  struct constant_coefficient<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
  template<typename Coefficients, typename NestedMatrix>
  struct constant_coefficient<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
    Coefficients::axes_only and constant_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<typename Coefficients, constant_matrix NestedMatrix> requires Coefficients::axes_only
  struct constant_coefficient<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
  template<typename Coefficients, typename NestedMatrix>
  struct constant_coefficient<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
    Coefficients::axes_only and constant_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_coefficient_v<NestedMatrix>> {};


  // -------------------------------- //
  //  constant_diagonal_coefficients  //
  // -------------------------------- //

  template<typename Scalar, auto constant>
  struct constant_diagonal_coefficient<ConstantMatrix<Scalar, constant, 1, 1>>
    : constant_coefficient_type<constant> {};


  template<typename Scalar, std::size_t dim>
  struct constant_diagonal_coefficient<ZeroMatrix<Scalar, dim, dim>>
    : constant_coefficient_type<short {0}, Scalar> {};


#ifdef __cpp_concepts
  template<constant_matrix NestedMatrix>
  struct constant_diagonal_coefficient<DiagonalMatrix<NestedMatrix>>
#else
  template<typename NestedMatrix>
  struct constant_diagonal_coefficient<DiagonalMatrix<NestedMatrix>, std::enable_if_t<constant_matrix<NestedMatrix>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix NestedMatrix, TriangleType triangle_type>
  struct constant_diagonal_coefficient<TriangularMatrix<NestedMatrix, triangle_type>>
#else
  template<typename NestedMatrix, TriangleType triangle_type>
  struct constant_diagonal_coefficient<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
    constant_diagonal_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<constant_matrix NestedMatrix>
  struct constant_diagonal_coefficient<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
#else
  template<typename NestedMatrix, TriangleType triangle_type>
  struct constant_diagonal_coefficient<TriangularMatrix<NestedMatrix, triangle_type>, std::enable_if_t<
    constant_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<constant_diagonal_matrix NestedMatrix, TriangleType storage_type>
  struct constant_diagonal_coefficient<SelfAdjointMatrix<NestedMatrix, storage_type>>
#else
  template<typename NestedMatrix, TriangleType storage_type>
  struct constant_diagonal_coefficient<SelfAdjointMatrix<NestedMatrix, storage_type>, std::enable_if_t<
    constant_diagonal_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<constant_matrix NestedMatrix>
  struct constant_diagonal_coefficient<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>>
#else
  template<typename NestedMatrix, TriangleType storage_type>
  struct constant_diagonal_coefficient<SelfAdjointMatrix<NestedMatrix, TriangleType::diagonal>, std::enable_if_t<
    constant_matrix<NestedMatrix>>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<typename Coefficients, constant_diagonal_matrix NestedMatrix> requires Coefficients::axes_only
  struct constant_diagonal_coefficient<ToEuclideanExpr<Coefficients, NestedMatrix>>
#else
  template<typename Coefficients, typename NestedMatrix>
  struct constant_diagonal_coefficient<ToEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
    constant_diagonal_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


#ifdef __cpp_concepts
  template<typename Coefficients, constant_diagonal_matrix NestedMatrix> requires Coefficients::axes_only
  struct constant_diagonal_coefficient<FromEuclideanExpr<Coefficients, NestedMatrix>>
#else
  template<typename Coefficients, typename NestedMatrix>
  struct constant_diagonal_coefficient<FromEuclideanExpr<Coefficients, NestedMatrix>, std::enable_if_t<
      constant_diagonal_matrix<NestedMatrix> and Coefficients::axes_only>>
#endif
    : constant_coefficient_type<constant_diagonal_coefficient_v<NestedMatrix>> {};


  // -------------------- //
  //  is_diagonal_matrix  //
  // -------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_diagonal_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim > 0) and (dim == 1 or constant == 0)> {};


  template<typename Scalar, std::size_t dim>
  struct is_diagonal_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim > 0)> {};


  template<typename NestedMatrix>
  struct is_diagonal_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_diagonal_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or triangle_type == TriangleType::diagonal> {};


  template<typename NestedMatrix, TriangleType storage_type>
  struct is_diagonal_matrix<SelfAdjointMatrix<NestedMatrix, storage_type>>
    : std::bool_constant<diagonal_matrix<NestedMatrix> or storage_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_diagonal_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and diagonal_matrix<NestedMatrix>> {};


  // ------------------------------ //
  //  is_lower_self_adjoint_matrix  //
  // ------------------------------ //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_lower_self_adjoint_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim > 0) and not complex_number<decltype(constant)>> {};


  template<typename Scalar, std::size_t dim>
  struct is_lower_self_adjoint_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim > 0)> {};


  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<DiagonalMatrix<NestedMatrix>>
    : std::bool_constant<not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_lower_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<storage_triangle != TriangleType::upper> {};


  template<typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::bool_constant<not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_self_adjoint_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_self_adjoint_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_self_adjoint_matrix<NestedMatrix>> {};


  // ------------------------------ //
  //  is_upper_self_adjoint_matrix  //
  // ------------------------------ //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_upper_self_adjoint_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim > 0) and not complex_number<decltype(constant)>> {};


  template<typename Scalar, std::size_t dim>
  struct is_upper_self_adjoint_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim > 0)> {};


  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<DiagonalMatrix<NestedMatrix>>
    : std::bool_constant<not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>> {};


  template<typename NestedMatrix, TriangleType storage_triangle>
  struct is_upper_self_adjoint_matrix<SelfAdjointMatrix<NestedMatrix, storage_triangle>>
    : std::bool_constant<storage_triangle != TriangleType::lower> {};


  template<typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<TriangularMatrix<NestedMatrix, TriangleType::diagonal>>
    : std::bool_constant<not complex_number<typename MatrixTraits<NestedMatrix>::Scalar>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_self_adjoint_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_self_adjoint_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_self_adjoint_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_lower_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_lower_triangular_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<(dim > 0) and (dim == 1 or constant == 0)> {};


  template<typename Scalar, std::size_t dim>
  struct is_lower_triangular_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim > 0)> {};


  template<typename NestedMatrix>
  struct is_lower_triangular_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_lower_triangular_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<lower_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::lower or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_lower_triangular_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and lower_triangular_matrix<NestedMatrix>> {};


  // ---------------------------- //
  //  is_upper_triangular_matrix  //
  // ---------------------------- //

  template<typename Scalar, auto constant, std::size_t dim>
  struct is_upper_triangular_matrix<ConstantMatrix<Scalar, constant, dim, dim>>
    : std::bool_constant<constant == 0 or dim == 1> {};


  template<typename Scalar, std::size_t dim>
  struct is_upper_triangular_matrix<ZeroMatrix<Scalar, dim, dim>>
    : std::bool_constant<(dim > 0)> {};


  template<typename NestedMatrix>
  struct is_upper_triangular_matrix<DiagonalMatrix<NestedMatrix>>
    : std::true_type {};


  template<typename NestedMatrix, TriangleType triangle_type>
  struct is_upper_triangular_matrix<TriangularMatrix<NestedMatrix, triangle_type>>
    : std::bool_constant<upper_triangular_matrix<NestedMatrix> or
        triangle_type == TriangleType::upper or triangle_type == TriangleType::diagonal> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<ToEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  template<typename Coefficients, typename NestedMatrix>
  struct is_upper_triangular_matrix<FromEuclideanExpr<Coefficients, NestedMatrix>>
    : std::bool_constant<Coefficients::axes_only and upper_triangular_matrix<NestedMatrix>> {};


  // ------------------- //
  //  is_self_contained  //
  // ------------------- //

  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns>
  struct is_self_contained<ConstantMatrix<Scalar, constant, rows, columns>> : std::true_type {};


  template<typename Scalar, std::size_t rows, std::size_t columns>
  struct is_self_contained<ZeroMatrix<Scalar, rows, columns>> : std::true_type {};


#ifdef __cpp_concepts
  template<typename T> requires
    eigen_diagonal_expr<T> or
    eigen_self_adjoint_expr<T> or
    eigen_triangular_expr<T> or
    to_euclidean_expr<T> or
    from_euclidean_expr<T>
  struct is_self_contained<T> : std::bool_constant<self_contained<nested_matrix_t<T>>> {};
#else
  template<typename T>
  struct is_self_contained<T, std::enable_if_t<
    eigen_diagonal_expr<T> or
    eigen_self_adjoint_expr<T> or
    eigen_triangular_expr<T> or
    to_euclidean_expr<T> or
    from_euclidean_expr<T>>> : std::bool_constant<self_contained<nested_matrix_t<T>>> {};
#endif


  // --------------------- //
  //  is_element_gettable  //
  // --------------------- //

#ifdef __cpp_concepts
  template<eigen_constant_expr T, std::size_t N>
  requires (N == 2) or (N == 1 and (column_vector<T> or row_vector<T>))
  struct is_element_gettable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<eigen_constant_expr<T> and
    ((N == 2) or (N == 1 and (column_vector<T> or row_vector<T>)))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<eigen_zero_expr T, std::size_t N>
  requires (N == 2) or (N == 1 and (column_vector<T> or row_vector<T>))
  struct is_element_gettable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<eigen_zero_expr<T> and
    ((N == 2) or (N == 1 and (column_vector<T> or row_vector<T>)))>>
#endif
    : std::true_type {};


#ifdef __cpp_concepts
  template<eigen_diagonal_expr T, std::size_t N>
  requires (N <= 2) and (element_gettable<nested_matrix_t<T>, 2> or element_gettable<nested_matrix_t<T>, 1>)
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<eigen_diagonal_expr<T> and (N <= 2)>>
    : std::bool_constant<element_gettable<nested_matrix_t<T>, 2> or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr T, std::size_t N>
  requires (element_gettable<nested_matrix_t<T>, 2> and
    (N == 2 or self_adjoint_triangle_type_of_v<T> == TriangleType::diagonal)) or element_gettable<nested_matrix_t<T>, 1>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<eigen_self_adjoint_expr<T>>>
    : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or self_adjoint_triangle_type_of_v<T> == TriangleType::diagonal)) or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<eigen_triangular_expr T, std::size_t N>
  requires (element_gettable<nested_matrix_t<T>, 2> and
    (N == 2 or triangle_type_of_v<T> == TriangleType::diagonal)) or element_gettable<nested_matrix_t<T>, 1>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<eigen_triangular_expr<T>>>
    : std::bool_constant<(element_gettable<nested_matrix_t<T>, 2> and
        (N == 2 or triangle_type_of_v<T> == TriangleType::diagonal)) or
        element_gettable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<to_euclidean_expr T, std::size_t N>
  requires element_gettable<nested_matrix_t<T>, N>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<to_euclidean_expr<T>>>
    : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::size_t N>
  requires element_gettable<nested_matrix_t<T>, N>
  struct is_element_gettable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_gettable<T, N, std::enable_if_t<from_euclidean_expr<T>>>
    : is_element_gettable<nested_matrix_t<T>, N> {};
#endif


  // --------------------- //
  //  is_element_settable  //
  // --------------------- //

#ifdef __cpp_concepts
  template<eigen_constant_expr T, std::size_t N>
  struct is_element_settable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<eigen_constant_expr<T>>>
#endif
    : std::false_type {};


#ifdef __cpp_concepts
  template<eigen_zero_expr T, std::size_t N>
  struct is_element_settable<T, N>
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<eigen_zero_expr<T>>>
#endif
    : std::false_type {};


#ifdef __cpp_concepts
  template<eigen_diagonal_expr T, std::size_t N>
  requires (N <= 2) and (element_settable<nested_matrix_t<T>, 2> or element_settable<nested_matrix_t<T>, 1>)
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<eigen_diagonal_expr<T> and (N <= 2)>>
    : std::bool_constant<element_settable<nested_matrix_t<T>, 2> or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr T, std::size_t N>
  requires (element_settable<nested_matrix_t<T>, 2> and
    (N == 2 or self_adjoint_triangle_type_of_v<T> == TriangleType::diagonal)) or element_settable<nested_matrix_t<T>, 1>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<eigen_self_adjoint_expr<T>>>
    : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or self_adjoint_triangle_type_of_v<T> == TriangleType::diagonal)) or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<eigen_triangular_expr T, std::size_t N>
  requires (element_settable<nested_matrix_t<T>, 2> and
    (N == 2 or triangle_type_of_v<T> == TriangleType::diagonal)) or element_settable<nested_matrix_t<T>, 1>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<eigen_triangular_expr<T>>>
    : std::bool_constant<(element_settable<nested_matrix_t<T>, 2> and
        (N == 2 or triangle_type_of_v<T> == TriangleType::diagonal)) or
        element_settable<nested_matrix_t<T>, 1>> {};
#endif


#ifdef __cpp_concepts
  template<to_euclidean_expr T, std::size_t N>
  requires MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<to_euclidean_expr<T> and
    MatrixTraits<T>::RowCoefficients::axes_only>>
    : is_element_settable<nested_matrix_t<T>, N> {};
#endif


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::size_t N>
  requires (MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>) or
    (to_euclidean_expr<nested_matrix_t<T>> and element_settable<nested_matrix_t<nested_matrix_t<T>>, N>)
  struct is_element_settable<T, N> : std::true_type {};
#else
  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<from_euclidean_expr<T> and
    (not to_euclidean_expr<nested_matrix_t<T>>)>>
    : std::bool_constant<MatrixTraits<T>::RowCoefficients::axes_only and element_settable<nested_matrix_t<T>, N>> {};

  template<typename T, std::size_t N>
  struct is_element_settable<T, N, std::enable_if_t<from_euclidean_expr<T> and
    to_euclidean_expr<nested_matrix_t<T>>>>
    : std::bool_constant<element_settable<nested_matrix_t<nested_matrix_t<T>>, N>> {};
#endif


  // ------------- //
  //  is_writable  //
  // ------------- //

#ifdef __cpp_concepts
  template<typename T>
  requires (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T> or
    to_euclidean_expr<T> or from_euclidean_expr<T>) and
    writable<nested_matrix_t<T>>
  struct is_writable<T> : std::true_type {};
#else
  template<typename T>
    struct is_writable<T, std::enable_if_t<
      (eigen_diagonal_expr<T> or eigen_self_adjoint_expr<T> or eigen_triangular_expr<T> or
      to_euclidean_expr<T> or from_euclidean_expr<T>) and
      writable<nested_matrix_t<T>>>> : std::true_type {};
#endif


  // ---------------------- //
  //  is_modifiable_native  //
  // ---------------------- //

  template<typename Scalar, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<ZeroMatrix<Scalar, rows, columns>, U>
    : std::false_type {};


  template<typename Scalar, auto constant, std::size_t rows, std::size_t columns, typename U>
  struct is_modifiable_native<ConstantMatrix<Scalar, constant, rows, columns>, U>
    : std::false_type {};


  template<typename N1, typename N2>
  struct is_modifiable_native<DiagonalMatrix<N1>, DiagonalMatrix<N2>>
    : std::bool_constant<modifiable<N1, N2>> {};


  template<typename NestedMatrix, typename U>
  struct is_modifiable_native<DiagonalMatrix<NestedMatrix>, U> : std::bool_constant<diagonal_matrix<U>> {};


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType storage_triangle, typename U> requires
    //(not self_adjoint_matrix<U>) or
    //(storage_triangle == TriangleType::diagonal and not diagonal_matrix<U>) or
    (eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix>) or
    (not eigen_self_adjoint_expr<U> and not modifiable<NestedMatrix, U>)
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, storage_triangle>, U> : std::false_type {};
#else
  template<typename N1, TriangleType t1, typename N2, TriangleType t2>
  struct is_modifiable_native<SelfAdjointMatrix<N1, t1>, SelfAdjointMatrix<N2, t2>>
    : std::bool_constant<modifiable<N1, N2>> {};

  template<typename NestedMatrix, TriangleType t, typename U>
  struct is_modifiable_native<SelfAdjointMatrix<NestedMatrix, t>, U>
    : std::bool_constant</*self_adjoint_matrix<U> and (t != TriangleType::diagonal or diagonal_matrix<U>) and
      */modifiable<NestedMatrix, U>> {};
#endif


#ifdef __cpp_concepts
  template<typename NestedMatrix, TriangleType triangle_type, typename U> requires
    //(not triangular_matrix<U>) or
    //(triangle_type == TriangleType::diagonal and not diagonal_matrix<U>) or
    (eigen_triangular_expr<U> and not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix>) or
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
      not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
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
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::dimensions == MatrixTraits<U>::rows and
      MatrixTraits<NestedMatrix>::columns == MatrixTraits<U>::columns and
      std::is_same_v<typename MatrixTraits<NestedMatrix>::Scalar, typename MatrixTraits<U>::Scalar>*/> {};
#endif


#ifdef __cpp_concepts
  template<typename C, typename NestedMatrix, typename U> requires
    (euclidean_expr<U> and (from_euclidean_expr<U> or
      not modifiable<NestedMatrix, typename MatrixTraits<U>::NestedMatrix> or
      not equivalent_to<C, typename MatrixTraits<U>::RowCoefficients>)) or
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
    : std::bool_constant<modifiable<NestedMatrix, native_matrix_t<NestedMatrix>>/* and C::euclidean_dimensions == MatrixTraits<U>::rows and
      MatrixTraits<NestedMatrix>::columns == MatrixTraits<U>::columns and
      std::is_same_v<typename MatrixTraits<NestedMatrix>::Scalar, typename MatrixTraits<U>::Scalar>*/> {};
#endif


} // namespace OpenKalman::internal

#endif //OPENKALMAN_EIGEN3_SPECIAL_MATRIX_TRAITS_HPP
