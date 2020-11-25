/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP

#include <type_traits>


/**
 * Namespace for all Eigen3 interface definitions.
 */
namespace OpenKalman::Eigen3
{
  // ---------------------------- //
  //    New Eigen matrix types    //
  // ---------------------------- //

  /**
   * \brief A self-adjoint matrix, based on an Eigen matrix.
   * \tparam NestedMatrix The Eigen matrix on which the self-adjoint matrix is based.
   * \tparam storage_triangle The triangle (TriangleType::upper or TriangleType::lower) in which the data is stored.
   */
  template<typename NestedMatrix, TriangleType storage_triangle = TriangleType::lower>
  struct SelfAdjointMatrix;


  /**
   * \brief A triangular matrix, based on an Eigen matrix.
   * \tparam NestedMatrix The Eigen matrix on which the triangular matrix is based.
   * \tparam triangle_type The triangle (TriangleType::upper or TriangleType::lower).
   */
  template<typename NestedMatrix, TriangleType triangle_type = TriangleType::lower>
  struct TriangularMatrix;


  /**
   * \brief A diagonal matrix, based on an Eigen matrix.
   * \note This has the same name as Eigen::DiagonalMatrix, and is intended as an improved replacement.
   * \tparam NestedMatrix A single-column matrix defining the diagonal.
   */
  template<typename NestedMatrix>
  struct DiagonalMatrix;


  /**
   * \brief A wrapper type for an Eigen zero matrix.
   * \note This is necessary because Eigen3 types do not distinguish between a zero matrix and a constant matrix.
   * \tparam NestedMatrix The Eigen matrix type on which the zero matrix is based. Only its shape is relevant.
   */
  template<typename NestedMatrix>
  struct ZeroMatrix;


  /**
   * \brief An expression that transforms coefficients into Euclidean space for proper wrapping.
   * \details This is the counterpart expression to ToEuclideanExpr.
   * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in
   * <code>B</code>.
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
   * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in
   * <code>B</code>.
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
    /*
     * Ultimate base class for new Eigen3 classes in OpenKalman.
     */
    template<typename Derived>
    struct Eigen3Base : Eigen::MatrixBase<Derived> { using type = Derived; };

    /*
     * Base class for all OpenKalman classes with a base that is an Eigen3 matrix.
     */
    template<typename Derived, typename Nested>
    struct Eigen3MatrixBase;

    /*
     * Base class for Covariance and SquareRootCovariance with a base that is an Eigen3 matrix.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename Nested>
#else
    template<typename Derived, typename Nested, typename Enable = void>
#endif
    struct Eigen3CovarianceBase;

  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
