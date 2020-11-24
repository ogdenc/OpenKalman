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

namespace OpenKalman
{

  // ---------------------------- //
  //    New Eigen matrix types    //
  // ---------------------------- //
  namespace Eigen3
  {
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
     * \brief An expression that transforms angular or other modular coefficients into Euclidean space, for proper wrapping.
     * \details This is the counterpart expression to ToEuclideanExpr.
     * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in <code>B</code>.
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
     * <code>FromEuclideanExpr<C, ToEuclideanExpr<C, B>></code> acts to wrap the angular/modular values in <code>B</code>.
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
    template<typename Arg>
    using IdentityMatrix = Eigen::CwiseNullaryOp<Eigen::internal::scalar_identity_op<typename Arg::Scalar>, Arg>;

  } // Eigen3


  // ---------------------------------------------------- //
  //    Eigen3-specific overloads for OpenKalman types    //
  // ---------------------------------------------------- //

  // ------------ //
  //    Matrix    //
  // ------------ //

#if FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE

  // Default for Eigen: the nested matrix will be an Eigen::Matrix of the appropriate size.
#ifdef __cpp_concepts
  template<
    coefficients RowCoefficients,
    coefficients ColumnCoefficients = RowCoefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
  requires
    (RowCoefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<NestedMatrix>::columns) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<
    typename RowCoefficients,
    typename ColumnCoefficients = RowCoefficients,
    typename NestedMatrix = Eigen::Matrix<double, RowCoefficients::size, ColumnCoefficients::size>>
#endif
  struct Matrix;


  /// If the arguments are a sequence of scalars, deduce a single-column matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Matrix(Args ...) -> Matrix<Axes<sizeof...(Args)>, Axis,
  Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ---------- //
  //    Mean    //
  // ---------- //

  // By default when using Eigen3, a Mean is an Eigen3 column vector corresponding to the Coefficients.
#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
  requires (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::size, 1>>
#endif
  struct Mean;


  /// If the arguments are a sequence of scalars, deduce a single-column mean with all Axis coefficients.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Mean(Args ...) -> Mean<Axes<sizeof...(Args)>,
  Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ------------------- //
  //    EuclideanMean    //
  // ------------------- //

#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
  requires
    (Coefficients::dimension == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix = Eigen::Matrix<double, Coefficients::dimension, 1>>
#endif
  struct EuclideanMean;


  /// If the arguments are a sequence of scalars, construct a single-column Euclidean mean.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  EuclideanMean(Args ...) -> EuclideanMean<OpenKalman::Axes<sizeof...(Args)>,
    Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>, sizeof...(Args), 1>>;


  // ---------------- //
  //    Covariance    //
  // ---------------- //

#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
  requires (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
#endif
  struct Covariance;


  /// If the arguments are a sequence of scalars, derive a square, self-adjoint matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args,
    std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  Covariance(Args ...) -> Covariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
  Eigen3::SelfAdjointMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
    internal::constexpr_sqrt(sizeof...(Args)), internal::constexpr_sqrt(sizeof...(Args))>>>;


  // --------------------- //
  //  SquareRootCovariance //
  // --------------------- //

#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_nestable NestedMatrix =
  Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>> requires
  (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix =
    Eigen3::SelfAdjointMatrix<Eigen::Matrix<double, Coefficients::size, Coefficients::size>>>
#endif
  struct SquareRootCovariance;


  /// If the arguments are a sequence of scalars, derive a square, lower triangular matrix.
#ifdef __cpp_concepts
  template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
  template<typename ... Args, std::enable_if_t<
    (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
  SquareRootCovariance(Args ...) -> SquareRootCovariance<Axes<internal::constexpr_sqrt(sizeof...(Args))>,
  Eigen3::TriangularMatrix<Eigen::Matrix<std::decay_t<std::common_type_t<Args...>>,
    OpenKalman::internal::constexpr_sqrt(sizeof...(Args)), OpenKalman::internal::constexpr_sqrt(sizeof...(Args))>>>;


#endif // FIRST_EIGEN_INTERFACE == OPENKALMAN_EIGEN3_INTERFACE


  namespace Eigen3
  {
    // ---------------------------------------- //
    //  Make functions for OpenKalman matrices  //
    // ---------------------------------------- //

    /// Make Mean from a list of coefficients.
#ifdef __cpp_concepts
    template<coefficients RowCoefficients, coefficients ColumnCoefficients = RowCoefficients, typename ... Args>
    requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename RowCoefficients, typename ColumnCoefficients = RowCoefficients, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Matrix(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = RowCoefficients::size;
      constexpr auto cols = ColumnCoefficients::size;
      static_assert(dim * cols == sizeof...(Args));
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /// Make Mean from a list of coefficients.
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Matrix(Args ... args)
    {
      using Coeffs = Axes<sizeof...(Args)>;
      return make_Matrix<Coeffs, Coefficients<Axis>>(args...);
    }


    /// Make Mean from a Scalar type and one or two sets of Coefficients.
#ifdef __cpp_concepts
    template<typename Scalar, coefficients RowCoefficients, coefficients ColumnCoefficients = RowCoefficients> requires
    std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename RowCoefficients, typename ColumnCoefficients = RowCoefficients,
      std::enable_if_t<std::is_arithmetic_v<Scalar> and
        coefficients<RowCoefficients> and coefficients<ColumnCoefficients>, int> = 0>
#endif
    auto make_Matrix()
    {
      using Mat = Eigen::Matrix<Scalar, RowCoefficients::size, ColumnCoefficients::size>;
      return Matrix<RowCoefficients, ColumnCoefficients, Mat>();
    }


    /// Make Mean from a list of coefficients, if Coefficients types are known.
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename ... Args> requires
    (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<coefficients<Coefficients> and
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Mean(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr std::size_t dim = Coefficients::size;
      static_assert(sizeof...(Args) % dim == 0);
      constexpr auto cols = sizeof...(Args) / dim;
      using Mat = Eigen::Matrix<Scalar, dim, cols>;
      return Mean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /// Make Mean from a list of coefficients, assuming that Coefficients types are all Axis.
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Mean(Args ... args)
    {
      return make_Mean<OpenKalman::Axes<sizeof...(Args)>>(args...);
    }


    /// Make a default Eigen3 Mean, based on a Scalar type, a set of Coefficients, and a number of columns.
#ifdef __cpp_concepts
    template<typename Scalar, coefficients Coefficients, std::size_t cols = 1> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
      std::is_arithmetic_v<Scalar> and coefficients<Coefficients>, int> = 0>
#endif
    auto make_Mean()
    {
      return Mean<Coefficients, Eigen::Matrix<Scalar, Coefficients::size, cols>>();
    }


    /// Make Euclidean mean from a list of coefficients.
#ifdef __cpp_concepts
    template<coefficients Coefficients, typename ... Args> requires
    (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename Coefficients, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_EuclideanMean(Args ... args) noexcept
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::dimension;
      static_assert(sizeof...(Args) % dim == 0);
      using Mat = Eigen::Matrix<Scalar, dim, 1>;
      return EuclideanMean<Coefficients, Mat>(MatrixTraits<Mat>::make(args...));
    }


    /// Make Mean from a list of coefficients.
#ifdef __cpp_concepts
    template<typename ... Args> requires (sizeof...(Args) > 0) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_EuclideanMean(Args ... args) noexcept
    {
      using Coefficients = OpenKalman::Axes<sizeof...(Args)>;
      return make_EuclideanMean<Coefficients>(args...);
    }


    /// Make self-contained EuclideanMean from a Scalar type, a set of Coefficients, and a number of columns.
#ifdef __cpp_concepts
    template<typename Scalar, coefficients Coefficients, std::size_t cols = 1> requires std::is_arithmetic_v<Scalar>
#else
    template<typename Scalar, typename Coefficients, std::size_t cols = 1, std::enable_if_t<
      std::is_arithmetic_v<Scalar> and coefficients<Coefficients>, int> = 0>
#endif
    auto make_EuclideanMean()
    {
      using Mat = Eigen::Matrix<Scalar, Coefficients::dimension, cols>;
      return Mean<Coefficients, Mat>();
    }


    /// Make a Covariance, based on a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type, typename ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<
      typename Coefficients, TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and coefficients < Coefficients>and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Covariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::size;
      static_assert(sizeof...(Args) == dim * dim);
      using Mat = Eigen::Matrix<Scalar, dim, dim>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type...>;
      using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
      return Covariance<Coefficients, B>(MatrixTraits<SA>::make(args...));
    }


    /// Make an axes-only covariance, based on a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<TriangleType ... triangle_type, typename ... Args> requires (sizeof...(Args) > 0) and
      (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<TriangleType ... triangle_type, typename ... Args, std::enable_if_t<(sizeof...(Args) > 0) and
      sizeof...(triangle_type) <= 1 and std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_Covariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      static_assert(sizeof...(Args) == dim * dim);
      using Coefficients = OpenKalman::Axes<dim>;
      return make_Covariance<Coefficients, triangle_type...>(args...);
    }


    /// Make default Covariance, based the size on the number of coefficients.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type> requires (sizeof...(triangle_type) <= 1)
#else
    template<typename Coefficients, TriangleType ... triangle_type, std::enable_if_t<
      sizeof...(triangle_type) <= 1 and coefficients < Coefficients>, int> = 0>
#endif
    auto make_Covariance()
    {
      using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
      using T = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      using SA = Eigen3::SelfAdjointMatrix<Mat, triangle_type...>;
      using B = std::conditional_t<sizeof...(triangle_type) == 1, T, SA>;
      return Covariance<Coefficients, B>();
    }


    /// Make SquareRootCovariance matrix using a list of coefficients in row-major order representing a triangular matrix.
    /// Only the coefficients in the lower-left corner are significant.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type, typename ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<typename Coefficients, TriangleType ... triangle_type, typename ... Args,
      std::enable_if_t<(sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and coefficients<Coefficients> and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_SquareRootCovariance(Args ... args)
    {
      using Scalar = std::common_type_t<Args...>;
      constexpr auto dim = Coefficients::size;
      static_assert(sizeof...(Args) == dim * dim);
      using Mat = Eigen::Matrix<Scalar, dim, dim>;
      using B = std::conditional_t<(sizeof...(triangle_type) == 1), // Is triangle type specified?
        typename MatrixTraits<Mat>::template TriangularBaseType<triangle_type...>,
        typename MatrixTraits<Mat>::template TriangularBaseType<TriangleType::lower>>; // lower-triangular self-adjoint, by default
      return SquareRootCovariance<Coefficients, B>(MatrixTraits<Mat>::make(args...));
    }

    /// Make an axes-only covariance, based on a list of coefficients in row-major order.
#ifdef __cpp_concepts
    template<TriangleType ... triangle_type, typename ... Args> requires
    (sizeof...(Args) > 0) and (sizeof...(triangle_type) <= 1) and (std::is_arithmetic_v<Args> and ...)
#else
    template<TriangleType ... triangle_type, typename ... Args, std::enable_if_t<
      (sizeof...(Args) > 0) and sizeof...(triangle_type) <= 1 and
      std::conjunction_v<std::is_arithmetic<Args>...>, int> = 0>
#endif
    auto make_SquareRootCovariance(Args ... args) noexcept
    {
      constexpr auto dim = OpenKalman::internal::constexpr_sqrt(sizeof...(Args));
      static_assert(sizeof...(Args) == dim * dim);
      using Coefficients = OpenKalman::Axes<dim>;
      return make_SquareRootCovariance<Coefficients, triangle_type...>(args...);
    }


    /// Make default Covariance, based the size on the number of coefficients.
#ifdef __cpp_concepts
    template<coefficients Coefficients, TriangleType ... triangle_type> requires (sizeof...(triangle_type) <= 1)
#else
    template<typename Coefficients, TriangleType ... triangle_type,
      std::enable_if_t<sizeof...(triangle_type) <= 1 and coefficients<Coefficients>, int> = 0>
#endif
    auto make_SquareRootCovariance()
    {
      using Mat = Eigen::Matrix<double, Coefficients::size, Coefficients::size>;
      using B = Eigen3::TriangularMatrix<Mat, triangle_type...>;
      return SquareRootCovariance<Coefficients, B>();
    }


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

  } // namespace Eigen3

} // namespace OpenKalman

#endif //OPENKALMAN_EIGEN3_FORWARD_DECLARATIONS_HPP
