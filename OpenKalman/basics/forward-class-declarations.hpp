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
  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

  /**
   * \brief A tensor or other matrix in which all elements are a constant scalar value known at compile time.
   * \details The constant scalar value is, in general, a set of coefficients reflecting a vector in a vector space
   * comprising the scalar value. For example,
   * - An integral or real constant will consist of a single value.
   * - A complex constant will consist of two values (real and imaginary parts).
   * - A quaternion constant will consist of four values, etc.
   * There must be a constructor for scalar_type_of_t<PatternMatrix> taking all the constants as arguments.
   * \tparam PatternMatrix An \ref native_eigen_general matrix having the size and shape of this matrix
   * \tparam constant The constant scalar value or coefficients within a vector space representing the constant scalar value.
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, auto...constant>
  requires std::constructible_from<scalar_type_of_t<PatternMatrix>, decltype(constant)...>
#else
  template<typename PatternMatrix, auto...constant>
#endif
  struct ConstantAdapter;


  // ------------------ //
  //  constant_adapter  //
  // ------------------ //

  namespace detail
  {
    template<typename T>
    struct is_constant_adapter : std::false_type {};

    template<typename PatternMatrix, auto...constant>
    struct is_constant_adapter<ConstantAdapter<PatternMatrix, constant...>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a ConstantAdapter.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept constant_adapter = detail::is_constant_adapter<std::decay_t<T>>::value;
#else
  constexpr bool constant_adapter = detail::is_constant_adapter<std::decay_t<T>>::value;
#endif


  // ------------- //
  //  ZeroAdapter  //
  // ------------- //

    /**
   * \brief A ConstantAdapter in which all elements are 0.
   * \detail This is an Eigen-specific version of ZeroMatrix
   * \tparam PatternMatrix An \ref native_eigen_general matrix having the size and shape of this matrix
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix>
#else
  template<typename PatternMatrix>
#endif
  using ZeroAdapter = ConstantAdapter<PatternMatrix, 0>;


  // ---------------------------------------- //
  //  pattern_matrix_of, pattern_matrix_of_t  //
  // ---------------------------------------- //

  /**
   * \brief The native matrix on which an OpenKalman matrix adapter is patterned.
   * \details If T has a nested matrix, the pattern matrix will be that nested matrix.
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct pattern_matrix_of;

  template<typename PatternMatrix, auto...constant>
  struct pattern_matrix_of<ConstantAdapter<PatternMatrix, constant...>> { using type = PatternMatrix; };


#ifdef __cpp_concepts
  template<has_nested_matrix T>
  struct pattern_matrix_of<T> { using type = nested_matrix_of_t<T>; };
#else
  template<typename T>
  struct pattern_matrix_of<T, std::enable_if_t<has_nested_matrix<T>>> { using type = nested_matrix_of_t<T>; };
#endif


  /**
   * \brief Helper template for pattern_matrix_of.
   */
  template<typename T>
  using pattern_matrix_of_t = typename pattern_matrix_of<std::decay_t<T>>::type;


  // ------------------------------------- //
  //  DiagonalMatrix, eigen_diagonal_expr  //
  // ------------------------------------- //

  /**
   * \brief A diagonal matrix.
   * \details The matrix is guaranteed to be diagonal. It is ::self_contained iff NestedMatrix is ::self_contained.
   * Implicit conversions are available from any \ref diagonal_matrix of compatible size.
   * \tparam NestedMatrix A \ref column_vector expression defining the diagonal elements.
   * Elements outside the diagonal are automatically 0.
   * \note This has the same name as Eigen::DiagonalMatrix, and is intended as a replacement.
   */
#ifdef __cpp_concepts
  template<indexible NestedMatrix> requires dimension_size_of_index_is<NestedMatrix, 1, 1, Likelihood::maybe>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalMatrix;


    namespace detail
    {
      template<typename T>
      struct is_eigen_diagonal_expr : std::false_type {};

      template<typename NestedMatrix>
      struct is_eigen_diagonal_expr<DiagonalMatrix<NestedMatrix>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a diagonal matrix based on the Eigen library (i.e., DiaginalMatrix).
     */
    template<typename T>
  #ifdef __cpp_concepts
    concept eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
  #else
    constexpr bool eigen_diagonal_expr = detail::is_eigen_diagonal_expr<std::decay_t<T>>::value;
  #endif


  // -------------------------------------------- //
  //  SelfAdjointMatrix, eigen_self_adjoint_expr  //
  // -------------------------------------------- //

  /**
   * \brief A hermitian matrix wrapper.
   * \details The matrix is guaranteed to be self-adjoint. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if storage_triangle is TriangleType::diagonal.
   * Implicit conversions are available from any \ref hermitian_matrix of compatible size.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the self-adjoint matrix is based.
   * \tparam storage_triangle The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. If the matrix is lower or upper triangular,
   * elements are mapped (as complex conjugates) from this selected triangle to the elements in the other triangle to
   * ensure that the matrix is hermitian. Also, any imaginary part of the diagonal elements is discarded.
   * If storage_triangle is TriangleType::diagonal, 0 is automatically mapped to each matrix element outside the
   * diagonal.
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, TriangleType storage_triangle =
      (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : TriangleType::lower)> requires
    (not constant_matrix<NestedMatrix> or imaginary_part(constant_coefficient_v<NestedMatrix>) == 0) and
    (not constant_diagonal_matrix<NestedMatrix> or imaginary_part(constant_diagonal_coefficient_v<NestedMatrix>) == 0) and
    (storage_triangle != TriangleType::none)
#else
  template<typename NestedMatrix, TriangleType storage_triangle =
    (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal : TriangleType::lower)>
#endif
  struct SelfAdjointMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_self_adjoint_expr : std::false_type {};

    template<typename NestedMatrix, TriangleType storage_triangle>
    struct is_eigen_self_adjoint_expr<SelfAdjointMatrix<NestedMatrix, storage_triangle>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a self-adjoint matrix based on the Eigen library (i.e., SelfAdjointMatrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#else
  constexpr bool eigen_self_adjoint_expr = detail::is_eigen_self_adjoint_expr<std::decay_t<T>>::value;
#endif


  // ----------------------------------------- //
  //  TriangularMatrix, eigen_triangular_expr  //
  // ----------------------------------------- //

  /**
   * \brief A triangular matrix.
   * \details The matrix is guaranteed to be triangular. It is ::self_contained iff NestedMatrix is ::self_contained.
   * It may \em also be a diagonal matrix if triangle_type is TriangleType::diagonal.
   * Implicit conversions are available from any \ref triangular_matrix of compatible size.
   * \tparam NestedMatrix A nested \ref square_matrix expression, on which the triangular matrix is based.
   * \tparam triangle_type The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. Instead, 0 is automatically mapped to each element
   * not within the selected triangle or diagonal, to ensure that the matrix is triangular.
   */
#ifdef __cpp_concepts
  template<square_matrix<Likelihood::maybe> NestedMatrix, TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
                                                                                        (upper_triangular_matrix<NestedMatrix> ? TriangleType::upper : TriangleType::lower))> requires
    (triangle_type != TriangleType::none)
#else
  template<typename NestedMatrix, TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
    (upper_triangular_matrix<NestedMatrix> ? TriangleType::upper : TriangleType::lower))>
#endif
  struct TriangularMatrix;


  namespace detail
  {
    template<typename T>
    struct is_eigen_triangular_expr : std::false_type {};

    template<typename NestedMatrix, TriangleType triangle_type>
    struct is_eigen_triangular_expr<TriangularMatrix<NestedMatrix, triangle_type>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a triangular matrix based on the Eigen library (i.e., TriangularMatrix).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#else
  constexpr bool eigen_triangular_expr = detail::is_eigen_triangular_expr<std::decay_t<T>>::value;
#endif


  // ----------------------- //
  //  Typed matrix adapters  //
  // ----------------------- //

  // -------------------------------------------------------- //
  //  FromEuclideanExpr, from_euclidean_expr, euclidean_expr  //
  // -------------------------------------------------------- //

  /**
   * \brief An expression that transforms angular or other modular coefficients back from Euclidean space.
   * \details This is the counterpart expression to ToEuclideanExpr.
   * \tparam TypedIndex The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typename NestedMatrix>
  requires (dynamic_index_descriptor<TypedIndex> == dynamic_rows<NestedMatrix>) and
    (not fixed_index_descriptor<TypedIndex> or euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not dynamic_index_descriptor<TypedIndex> or
      std::same_as<typename TypedIndex::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct FromEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_from_euclidean_expr : std::false_type {};

    template<typename TypedIndex, typename NestedMatrix>
    struct is_from_euclidean_expr<FromEuclideanExpr<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients from Euclidean space (i.e., FromEuclideanExpr).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#else
  constexpr bool from_euclidean_expr = detail::is_from_euclidean_expr<std::decay_t<T>>::value;
#endif


  // ------------------------------------ //
  //  ToEuclideanExpr, to_euclidean_expr  //
  // ------------------------------------ //

  /**
   * \brief An expression that transforms coefficients into Euclidean space for proper wrapping.
   * \details This is the counterpart expression to FromEuclideanExpr.
   * \tparam TypedIndex The coefficient types.
   * \tparam NestedMatrix The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typename NestedMatrix> requires (not from_euclidean_expr<NestedMatrix>) and
    (dynamic_index_descriptor<TypedIndex> == dynamic_rows<NestedMatrix>) and
    (not fixed_index_descriptor<TypedIndex> or dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not dynamic_index_descriptor<TypedIndex> or
      std::same_as<typename TypedIndex::Scalar, scalar_type_of_t<NestedMatrix>>)
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct ToEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_to_euclidean_expr : std::false_type {};

    template<typename TypedIndex, typename NestedMatrix>
    struct is_to_euclidean_expr<ToEuclideanExpr<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients to Euclidean space (i.e., ToEuclideanExpr).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#else
  constexpr bool to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is either \ref to_euclidean_expr or \ref from_euclidean_expr.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_expr = to_euclidean_expr<T> or from_euclidean_expr<T>;
#else
  constexpr bool euclidean_expr = from_euclidean_expr<T> or to_euclidean_expr<T>;
#endif


  /**
   * \brief A matrix with typed rows and columns.
   * \details It is a wrapper for a native matrix type from a supported matrix library such as Eigen.
   * The matrix can be thought of as a tests from X to Y, where the coefficients for each of X and Y are typed.
   * Example declarations:
   * - <code>Matrix<TypedIndex<Axis, Axis, angle::Radians>, TypedIndex<Axis, Axis>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * - <code>Matrix<double, TypedIndex<Axis, Axis, angle::Radians>, TypedIndex<Axis, Axis>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * \tparam RowCoefficients A set of \ref OpenKalman::coefficients "coefficients" (e.g., Axis, Spherical, etc.)
   * corresponding to the rows.
   * \tparam ColumnCoefficients Another set of \ref OpenKalman::coefficients "coefficients" corresponding
   * to the columns.
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor RowCoefficients, fixed_index_descriptor ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
    (dimension_size_of_v<ColumnCoefficients> == column_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_index_descriptor<RowCoefficients> == dynamic_rows<NestedMatrix>) and
    (dynamic_index_descriptor<ColumnCoefficients> == dynamic_columns<NestedMatrix>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix;


  namespace internal
  {
    template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
    struct is_matrix<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A set of one or more column vectors, each representing a statistical mean.
   * \details Unlike OpenKalman::Matrix, the columns of a Mean are untyped. When a Mean is converted to an
   * OpenKalman::Matrix, the columns are assigned type Axis.
   * Example declaration:
   * <code>Mean<TypedIndex<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * \tparam TypedIndex Coefficient types of the mean (e.g., Axis, Polar).
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor RowCoefficients, typed_matrix_nestable NestedMatrix> requires
  (dimension_size_of_v<RowCoefficients> == row_dimension_of_v<NestedMatrix>) and
  (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean;


  namespace internal
  {
    template<typename TypedIndex, typename NestedMatrix>
    struct is_mean<Mean<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Similar to a Mean, but the coefficients are transformed into Euclidean space, based on their type.
   * \details Means containing angles should be converted to EuclideanMean before taking an average or weighted average.
   * Example declaration:
   * <code>EuclideanMean<TypedIndex<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 4, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with the last two of the dimensions representing the angle::Radians coefficient
   * transformed to x and y locations on a unit circle associated with the angle::Radians-type coefficient.
   * \tparam TypedIndex A set of coefficients (e.g., Axis, angle::Radians, Polar, etc.)
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, typed_matrix_nestable NestedMatrix> requires
  (euclidean_dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct EuclideanMean;


  namespace internal
  {
    template<typename TypedIndex, typename NestedMatrix>
    struct is_euclidean_mean<EuclideanMean<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A self-adjoint Covariance matrix.
   * \details The coefficient types for the rows are the same as for the columns.
   * \tparam TypedIndex Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is triangular, the native matrix will be multiplied by its transpose
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, covariance_nestable NestedMatrix> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct Covariance;


  namespace internal
  {
    template<typename TypedIndex, typename NestedMatrix>
    struct is_self_adjoint_covariance<Covariance<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * \details If S is a SquareRootCovariance, S*transpose(S) is a Covariance.
   * If NestedMatrix is triangular, the SquareRootCovariance has the same triangle type (upper or lower). If NestedMatrix
   * is self-adjoint, the triangle type of SquareRootCovariance is considered either upper ''or'' lower.
   * \tparam TypedIndex Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is self-adjoint, the native matrix will be Cholesky-factored
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<fixed_index_descriptor TypedIndex, covariance_nestable NestedMatrix> requires
    (dimension_size_of_v<TypedIndex> == row_dimension_of_v<NestedMatrix>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and scalar_type<scalar_type_of_t<NestedMatrix>>
#else
  template<typename TypedIndex, typename NestedMatrix>
#endif
  struct SquareRootCovariance;


  namespace internal
  {
    template<typename TypedIndex, typename NestedMatrix>
    struct is_triangular_covariance<SquareRootCovariance<TypedIndex, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A Gaussian distribution, defined in terms of a Mean and a Covariance.
   * \tparam TypedIndex Coefficient types.
   * \tparam MeanNestedMatrix The underlying native matrix for the Mean.
   * \tparam CovarianceNestedMatrix The underlying native matrix (triangular or self-adjoint) for the Covariance.
   * \tparam random_number_engine A random number engine compatible with the c++ standard library (e.g., std::mt19937).
   * \todo Change to std::mt19937_64 ?
   */
#ifdef __cpp_concepts
  template<
    fixed_index_descriptor TypedIndex,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine = std::mt19937> requires
      (row_dimension_of_v<MeanNestedMatrix> == row_dimension_of_v<CovarianceNestedMatrix>) and
      (column_dimension_of_v<MeanNestedMatrix> == 1) and
      (std::is_same_v<scalar_type_of_t<MeanNestedMatrix>,
        scalar_type_of_t<CovarianceNestedMatrix>>)
#else
  template<
    typename TypedIndex,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine = std::mt19937>
#endif
  struct GaussianDistribution;


  namespace internal
  {
    template<typename TypedIndex, typename MeanNestedMatrix, typename CovarianceNestedMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<TypedIndex, MeanNestedMatrix, CovarianceNestedMatrix, re>>
      : std::true_type {};
  }


  // ================== //
  //  internal classes  //
  // ================== //

  namespace internal
  {
    /**
     * \internal
     * \brief A base class within a particular library
     * \details This is defined by MatrixTraits<std::decay_t<PatternMatrix>>::template MatrixBaseFrom<Derived>
     * \tparam Derived A derived class, such as an adapter.
     * \tparam PatternMatrix A class within a particular library
     */
#if defined(__cpp_concepts) and OPENKALMAN_CPP_FEATURE_CONCEPTS
    template<typename Derived, typename PatternMatrix>
#else
    template<typename Derived, typename PatternMatrix, typename = void>
#endif
    struct library_base;


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
     * \tparam TypedIndex The \ref OpenKalman::coefficients "coefficients" representing the rows and columns of the matrix.
     */
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix, fixed_index_descriptor...TypedIndex>
    requires (not std::is_rvalue_reference_v<NestedMatrix>) and (sizeof...(TypedIndex) <= 2)
#else
    template<typename Derived, typename NestedMatrix, typename...TypedIndex>
#endif
    struct TypedMatrixBase;


    /**
     * \internal
     * \brief An interface to a matrix, to be used for getting and setting the individual matrix elements.
     * \tparam settable Whether the matrix elements can be set (as opposed to being read-only).
     * \tparam Scalar the scalar type of the elements.
     */
    template<bool settable, typename Scalar = double>
    struct ElementAccessor;


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


  } // namespace internal

} // OpenKalman

#endif //OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
