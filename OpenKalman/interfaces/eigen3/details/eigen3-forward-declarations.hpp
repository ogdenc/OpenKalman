/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \dir details
 * \brief Support files for the Eigen3 interface
 *
 * \file
 * \brief Forward declarations for OpenKalman's Eigen3 interface.
 */

#ifndef OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP

#include <type_traits>


/**
 * \namespace OpenKalman::Eigen3
 * \brief Namespace for all Eigen3 interface definitions.
 *
 * \internal
 * \namespace OpenKalman::Eigen3::internal
 * \brief Namespace for definitions internal to the Eigen3 interface library.
 *
 * \namespace Eigen
 * \brief Eigen3's native namespace.
 *
 * \namespace Eigen::internal
 * \brief Eigen3's native namespace for internal definitions.
 */


namespace OpenKalman::Eigen3
{
  // ---------------------------- //
  //    New Eigen matrix types    //
  // ---------------------------- //

  /**
   * \brief A self-adjoint matrix.
   * \details The matrix is guaranteed to be self-adjoint. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if storage_triangle is TriangleType::diagonal.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the self-adjoint matrix is based.
   * \tparam storage_triangle The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. If the matrix is lower or upper triangular,
   * elements are mapped from this selected triangle to the elements in the other triangle to ensure that the matrix
   * is self-adjoint. If the matrix is diagonal, 0 is automatically mapped to each matrix element outside the diagonal.
   */
#ifdef __cpp_concepts
  template<square_matrix NestedMatrix, TriangleType storage_triangle = TriangleType::lower>
#else
  template<typename NestedMatrix, TriangleType storage_triangle = TriangleType::lower>
#endif
  struct SelfAdjointMatrix;


  /**
   * \brief A triangular matrix.
   * \details The matrix is guaranteed to be triangular. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if triangle_type is TriangleType::diagonal.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the triangular matrix is based.
   * \tparam triangle_type The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. Instead, 0 is automatically mapped to each element
   * not within the selected triangle or diagonal, to ensure that the matrix is triangular.
   */
#ifdef __cpp_concepts
  template<square_matrix NestedMatrix, TriangleType triangle_type = TriangleType::lower>
#else
  template<typename NestedMatrix, TriangleType triangle_type = TriangleType::lower>
#endif
  struct TriangularMatrix;


  /**
   * \brief A diagonal matrix.
   * \details The matrix is guaranteed to be diagonal. It is ::self_contained iff NestedMatrix is ::self_contained.
   * \tparam NestedMatrix A \ref column_vector expression defining the diagonal elements.
   * Elements outside the diagonal are automatically 0.
   * \note This has the same name as Eigen::DiagonalMatrix, and is intended as a replacement.
   */
#ifdef __cpp_concepts
  template<column_vector NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix;


  /**
   * \brief A matrix in which all elements are automatically 0.
   * \note This is necessary because Eigen3 types do not distinguish between a zero matrix and a constant matrix.
   * \tparam Scalar The scalar type.
   * \tparam rows The number of rows.
   * \tparam columns The number of columns.
   */
  template<typename Scalar, std::size_t rows, std::size_t columns = 1>
  struct ZeroMatrix;


  /**
   * \brief An expression that transforms coefficients into Euclidean space for proper wrapping.
   * \details This is the counterpart expression to FromEuclideanExpr.
   * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, M>></code> acts to wrap the angular/modular values in
   * <code>M</code>.
   * \tparam Coefficients The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
    requires (MatrixTraits<NestedMatrix>::dimension == Coefficients::size)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
#endif
  struct ToEuclideanExpr;


  /**
   * \brief An expression that transforms angular or other modular coefficients back from Euclidean space.
   * \details This is the counterpart expression to ToEuclideanExpr.
   * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, M>></code> acts to wrap the angular/modular values in
   * <code>M</code>.
   * \tparam Coefficients The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
    requires (MatrixTraits<NestedMatrix>::dimension == Coefficients::dimension)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
#endif
  struct FromEuclideanExpr;


  /**
   * \brief An alias for the Eigen identity matrix.
   */
  template<typename NestedMatrix>
  using IdentityMatrix =
    Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename NestedMatrix::Scalar>, NestedMatrix>;


  namespace internal
  {
    /**
     * \internal
     * \brief Ultimate base for matrix classes in OpenKalman.
     */
    template<typename Derived>
    struct Eigen3Base : Eigen::MatrixBase<Derived> {};


    /*
     * \internal
     * \brief Penultimate base for matrix classes in OpenKalman.
     */
    template<typename Derived, typename Nested>
    struct Eigen3MatrixBase;

  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
