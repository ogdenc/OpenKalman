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
 * \brief Forward declarations for OpenKalman classes and related traits.
 */

#ifndef OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
#define OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP

#include <type_traits>
#include <random>

namespace OpenKalman
{
  /**
   * \brief A matrix with typed rows and columns.
   * \details It is a wrapper for a native matrix type from a supported matrix library such as Eigen.
   * The matrix can be thought of as a tests from X to Y, where the coefficients for each of X and Y are typed.
   * Example declarations:
   * - <code>Matrix<Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * - <code>Matrix<double, Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * \tparam RowCoefficients A set of \ref OpenKalman::coefficients "coefficients" (e.g., Axis, Spherical, etc.)
   * corresponding to the rows.
   * \tparam ColumnCoefficients Another set of \ref OpenKalman::coefficients "coefficients" corresponding
   * to the columns.
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (RowCoefficients::dimensions == row_extent_of_v<NestedMatrix>) and
    (ColumnCoefficients::dimensions == column_extent_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_coefficients<RowCoefficients> == dynamic_rows<NestedMatrix>) and
    (dynamic_coefficients<ColumnCoefficients> == dynamic_columns<NestedMatrix>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix;


  namespace internal
  {
    template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
    struct is_matrix<OpenKalman::Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A set of one or more column vectors, each representing a statistical mean.
   * \details Unlike OpenKalman::Matrix, the columns of a Mean are untyped. When a Mean is converted to an
   * OpenKalman::Matrix, the columns are assigned type Axis.
   * Example declaration:
   * <code>Mean<Coefficients<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * \tparam Coefficients Coefficient types of the mean (e.g., Axis, Polar).
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, typed_matrix_nestable NestedMatrix> requires
  (RowCoefficients::dimensions == row_extent_of_v<NestedMatrix>) and
  (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean;


  namespace internal
  {
    template<typename Coefficients, typename NestedMatrix>
    struct is_mean<Mean<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Similar to a Mean, but the coefficients are transformed into Euclidean space, based on their type.
   * \details Means containing angles should be converted to EuclideanMean before taking an average or weighted average.
   * Example declaration:
   * <code>EuclideanMean<Coefficients<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 4, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with the last two of the dimensions representing the angle::Radians coefficient
   * transformed to x and y locations on a unit circle associated with the angle::Radians-type coefficient.
   * \tparam Coefficients A set of coefficients (e.g., Axis, angle::Radians, Polar, etc.)
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable NestedMatrix> requires
  (Coefficients::euclidean_dimensions == row_extent_of_v<NestedMatrix>) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct EuclideanMean;


  namespace internal
  {
    template<typename Coefficients, typename NestedMatrix>
    struct is_euclidean_mean<EuclideanMean<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A self-adjoint Covariance matrix.
   * \details The coefficient types for the rows are the same as for the columns.
   * \tparam Coefficients Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is triangular, the native matrix will be multiplied by its transpose
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix> requires
    (Coefficients::dimensions == row_extent_of_v<NestedMatrix>) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct Covariance;


  namespace internal
  {
    template<typename Coefficients, typename NestedMatrix>
    struct is_self_adjoint_covariance<Covariance<Coefficients, NestedMatrix>> : std::true_type {};

    //template<typename Coefficients, typename NestedMatrix>
    //struct is_upper_self_adjoint_matrix<Covariance<Coefficients, NestedMatrix>>
    //  : std::bool_constant<upper_self_adjoint_matrix<NestedMatrix> or upper_triangular_matrix<NestedMatrix>> {};

    //template<typename Coefficients, typename NestedMatrix>
    //struct is_lower_self_adjoint_matrix<Covariance<Coefficients, NestedMatrix>>
    //  : std::bool_constant<lower_self_adjoint_matrix<NestedMatrix> or lower_triangular_matrix<NestedMatrix>> {};
  }


  /**
   * \brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * \details If S is a SquareRootCovariance, S*transpose(S) is a Covariance.
   * If NestedMatrix is triangular, the SquareRootCovariance has the same triangle type (upper or lower). If NestedMatrix
   * is self-adjoint, the triangle type of SquareRootCovariance is considered either upper ''or'' lower.
   * \tparam Coefficients Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is self-adjoint, the native matrix will be Cholesky-factored
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix> requires
    (Coefficients::dimensions == row_extent_of_v<NestedMatrix>) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct SquareRootCovariance;


  namespace internal
  {
    template<typename Coefficients, typename NestedMatrix>
    struct is_triangular_covariance<SquareRootCovariance<Coefficients, NestedMatrix>> : std::true_type {};

    //template<typename Coefficients, typename NestedMatrix>
    //struct is_upper_triangular_matrix<SquareRootCovariance<Coefficients, NestedMatrix>>
    //  : std::bool_constant<upper_triangular_matrix<NestedMatrix> or upper_self_adjoint_matrix<NestedMatrix>> {};

    //template<typename Coefficients, typename NestedMatrix>
    //struct is_lower_triangular_matrix<SquareRootCovariance<Coefficients, NestedMatrix>>
    //  : std::bool_constant<lower_triangular_matrix<NestedMatrix> or lower_self_adjoint_matrix<NestedMatrix>> {};
  }


  /**
   * \brief A Gaussian distribution, defined in terms of a Mean and a Covariance.
   * \tparam Coefficients Coefficient types.
   * \tparam MeanNestedMatrix The underlying native matrix for the Mean.
   * \tparam CovarianceNestedMatrix The underlying native matrix (triangular or self-adjoint) for the Covariance.
   * \tparam random_number_engine A random number engine compatible with the c++ standard library (e.g., std::mt19937).
   * \todo Change to std::mt19937_64 ?
   */
#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine = std::mt19937> requires
      (row_extent_of_v<MeanNestedMatrix> == row_extent_of_v<CovarianceNestedMatrix>) and
      (column_extent_of_v<MeanNestedMatrix> == 1) and
      (std::is_same_v<scalar_type_of_t<MeanNestedMatrix>,
        scalar_type_of_t<CovarianceNestedMatrix>>)
#else
  template<
    typename Coefficients,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine = std::mt19937>
#endif
  struct GaussianDistribution;


  namespace internal
  {
    template<typename Coefficients, typename MeanNestedMatrix, typename CovarianceNestedMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<Coefficients, MeanNestedMatrix, CovarianceNestedMatrix, re>>
      : std::true_type {};
  }


  namespace internal
  {
    /**
     * \internal
     * \brief Ultimate base of typed matrices and covariance matrices.
     * \tparam Derived The fully derived matrix type.
     * \tparam NestedMatrix The nested native matrix, which can be const or an lvalue reference, or both, or neither.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix> requires (not std::is_rvalue_reference_v<NestedMatrix>)
#else
    template<typename Derived, typename NestedMatrix>
#endif
    struct MatrixBase;


    /**
     * \internal
     * \brief Base class for means or matrices.
     * \tparam Derived The derived class (e.g., Matrix, Mean, EuclideanMean).
     * \tparam NestedMatrix The nested matrix.
     * \tparam Coefficients The \ref OpenKalman::coefficients "coefficients" representing the rows and columns of the matrix.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix, coefficients...Coefficients>
    requires (not std::is_rvalue_reference_v<NestedMatrix>) and (sizeof...(Coefficients) <= 2)
#else
    template<typename Derived, typename NestedMatrix, typename...Coefficients>
#endif
    struct TypedMatrixBase;


    /**
     * \internal
     * \brief Base of Covariance and SquareRootCovariance classes.
     * \tparam Derived The fully derived covariance type.
     * \tparam NestedMatrix The nested native matrix, which can be const or an lvalue reference, or both, or neither.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix>
#else
    template<typename Derived, typename NestedMatrix, typename = void>
#endif
    struct CovarianceBase;


    /**
     * \internal
     * \brief Implementations for Covariance and SquareRootCovariance classes.
     * \tparam Derived The fully derived covariance type.
     * \tparam NestedMatrix The nested native matrix, which can be const or an lvalue reference, or both, or neither.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix>
#else
    template<typename Derived, typename NestedMatrix>
#endif
    struct CovarianceImpl;


    /**
     * \internal
     * \brief An interface to a matrix, to be used for getting and setting the individual matrix elements.
     * \tparam settable Whether the matrix elements can be set (as opposed to being read-only).
     * \tparam Scalar the scalar type of the elements.
     */
    template<bool settable, typename Scalar = double>
    struct ElementAccessor;


  } // namespace internal


} // OpenKalman

#endif //OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
