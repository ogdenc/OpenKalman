/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2024 Christopher Lee Ogden <ogden@gatech.edu>
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
#include "basics/traits/traits.hpp"

namespace OpenKalman
{
  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

  /**
   * \brief A tensor or other matrix in which all elements are a constant scalar value.
   * \details The constant value can be \ref value::fixed "static" (known at compile time), or
   * \ref value::dynamic "dynamic" (known only at runtime).
   * Examples:
   * \code
   * using T = Eigen::Matrix<double, 3, 2>; // A 3-by-2 matrix of scalar-type double.
   * ConstantAdapter<T> c1 {3.0}; // Construct a 3-by-2 double constant of shape T with value 3.0 (known at runtime).
   * ConstantAdapter<T, int> c2 {3}; // Construct a 3-by-2 int constant of shape T with value 3 (known at runtime).
   * ConstantAdapter<T, int, 1> c3; // Construct a 3-by-2 int constant of shape T with value 1 (known at compile time).
   * ConstantAdapter<T, double, 1> c4; // Construct a 3-by-2 double constant of shape T with value 1.0 (known at compile time).
   * ConstantAdapter<T, std::integral_constant<int, 1>> c5; // Construct a 3-by-2 int constant of shape T with value 1 (known at compile time).
   * ConstantAdapter<T, std::complex<double>> c6 {std::complex<double>{4, 5}}; // Construct a 3-by-2 complex constant of shape T and value 4.0 + 5.0i (known at runtime).
   * ConstantAdapter<T, std::complex<double>, 4, 5> c7; // Construct a 3-by-2 A complex constant of shape T and value 4.0 + 5.0i (known at compile time).
   * \endcode
   * \tparam PatternMatrix An \ref indexible object reflecting the size and shape of the constant object
   * \tparam Scalar A \ref value::number reflecting the type of the constant
   * \tparam constant Optional parameters for constructing Scalar at compile time.
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, value::scalar Scalar = scalar_type_of_t<PatternMatrix>, auto...constant>
    requires (sizeof...(constant) == 0) or requires { Scalar {constant...}; }
#else
  template<typename PatternMatrix, typename Scalar = scalar_type_of_t<PatternMatrix>, auto...constant>
#endif
  struct ConstantAdapter;


  // ------------------ //
  //  constant_adapter  //
  // ------------------ //

  namespace detail
  {
    template<typename T>
    struct is_constant_adapter : std::false_type {};

    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct is_constant_adapter<ConstantAdapter<PatternMatrix, Scalar, constant...>> : std::true_type {};
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
   * \tparam PatternObject An indexible object, in some library, defining the shape of the resulting zero object
   */
#ifdef __cpp_concepts
  template<indexible PatternObject, value::number Scalar = scalar_type_of_t<PatternObject>>
#else
  template<typename PatternObject, typename Scalar = scalar_type_of_t<PatternObject>>
#endif
  using ZeroAdapter = ConstantAdapter<PatternObject, Scalar, 0>;


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

  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct pattern_matrix_of<ConstantAdapter<PatternMatrix, Scalar, constant...>> { using type = PatternMatrix; };


#ifdef __cpp_concepts
  template<has_nested_object T>
  struct pattern_matrix_of<T> { using type = nested_object_of_t<T>; };
#else
  template<typename T>
  struct pattern_matrix_of<T, std::enable_if_t<has_nested_object<T>>> { using type = nested_object_of_t<T>; };
#endif


  /**
   * \brief Helper template for pattern_matrix_of.
   */
  template<typename T>
  using pattern_matrix_of_t = typename pattern_matrix_of<std::decay_t<T>>::type;


  // ------------------------------------------ //
  //  DiagonalAdapter, internal::diagonal_expr  //
  // ------------------------------------------ //

  /**
   * \brief An adapter for creating a diagonal matrix or tensor.
   * \details The matrix is guaranteed to be diagonal.
   * Implicit conversions are available from any \ref diagonal_matrix of compatible size.
   * \tparam NestedMatrix A column vector expression defining the diagonal elements.
   * indexible_object_traits outside the diagonal are automatically 0.
   */
#ifdef __cpp_concepts
  template<vector<0, Applicability::permitted> NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct DiagonalAdapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_diagonal_expr : std::false_type {};

      template<typename NestedMatrix>
      struct is_diagonal_expr<DiagonalAdapter<NestedMatrix>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a diagonal matrix based on the Eigen library (i.e., DiaginalMatrix).
     */
    template<typename T>
#ifdef __cpp_concepts
    concept diagonal_expr = detail::is_diagonal_expr<std::decay_t<T>>::value;
#else
    constexpr bool diagonal_expr = detail::is_diagonal_expr<std::decay_t<T>>::value;
#endif
  } // namespace internal


  // -------------------------------------------- //
  //  HermitianAdapter, internal::hermitian_expr  //
  // -------------------------------------------- //

  /**
   * \brief A hermitian matrix wrapper.
   * \details The matrix is guaranteed to be hermitian.
   * Implicit conversions are available from any \ref hermitian_matrix of compatible size.
   * \tparam NestedMatrix A nested \ref square_shaped expression, on which the self-adjoint matrix is based.
   * \tparam storage_triangle The HermitianAdapterType (\ref HermitianAdapterType::lower "lower" or
   * \ref HermitianAdapterType::upper "upper") in which the data is stored.
   * Matrix elements outside this triangle/diagonal are ignored. If the matrix is lower or upper triangular,
   * elements are mapped (as complex conjugates) from this selected triangle to the elements in the other triangle to
   * ensure that the matrix is hermitian. Also, any imaginary part of the diagonal elements is discarded.
   * If storage_triangle is TriangleType::diagonal, 0 is automatically mapped to each matrix element outside the
   * diagonal.
   */
#ifdef __cpp_concepts
  template<square_shaped<Applicability::permitted> NestedMatrix, HermitianAdapterType storage_triangle =
      triangular_matrix<NestedMatrix, TriangleType::diagonal> ? HermitianAdapterType::lower :
      triangular_matrix<NestedMatrix, TriangleType::upper> ? HermitianAdapterType::upper : HermitianAdapterType::lower> requires
    (index_count_v<NestedMatrix> <= 2) and
    (storage_triangle == HermitianAdapterType::lower or storage_triangle == HermitianAdapterType::upper) and
    (not constant_matrix<NestedMatrix> or value::not_complex<constant_coefficient<NestedMatrix>>) and
    (not constant_diagonal_matrix<NestedMatrix> or value::not_complex<constant_diagonal_coefficient<NestedMatrix>>) and
    (not triangular_matrix<NestedMatrix, TriangleType::any> or triangular_matrix<NestedMatrix, static_cast<TriangleType>(storage_triangle)>)
#else
  template<typename NestedMatrix, HermitianAdapterType storage_triangle =
    triangular_matrix<NestedMatrix, TriangleType::diagonal> ? HermitianAdapterType::lower :
    triangular_matrix<NestedMatrix, TriangleType::upper> ? HermitianAdapterType::upper : HermitianAdapterType::lower>
#endif
  struct HermitianAdapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_hermitian_expr : std::false_type {};

      template<typename NestedMatrix, HermitianAdapterType storage_triangle>
      struct is_hermitian_expr<HermitianAdapter<NestedMatrix, storage_triangle>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a self-adjoint matrix based on the Eigen library (i.e., HermitianAdapter).
     */
    template<typename T>
#ifdef __cpp_concepts
    concept hermitian_expr = detail::is_hermitian_expr<std::decay_t<T>>::value;
#else
    constexpr bool hermitian_expr = detail::is_hermitian_expr<std::decay_t<T>>::value;
#endif
  } // namespace internal


  // ---------------------------------------------- //
  //  TriangularAdapter, internal::triangular_expr  //
  // ---------------------------------------------- //

  /**
   * \brief A \ref triangular_adapter, where components above or below the diagonal (or both) are zero.
   * \details The matrix may be a diagonal matrix if triangle_type is TriangleType::diagonal.
   * Implicit conversions are available from any \ref triangular_matrix of compatible size.
   * \tparam NestedMatrix A nested matrix on which the triangular matrix is based. Components above or below the diagonal
   * (or both) are ignored and will read as zero.
   * \tparam triangle_type The TriangleType (\ref TriangleType::lower "lower", \ref TriangleType::upper "upper", or
   * \ref TriangleType::diagonal "diagonal") in which the data is stored.
   */
#ifdef __cpp_concepts
  template<
    square_shaped<Applicability::permitted> NestedMatrix,
    TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
      (triangular_matrix<NestedMatrix, TriangleType::upper> ? TriangleType::upper : TriangleType::lower))>
    requires (index_count_v<NestedMatrix> <= 2)
#else
  template<typename NestedMatrix, TriangleType triangle_type = (diagonal_matrix<NestedMatrix> ? TriangleType::diagonal :
    (triangular_matrix<NestedMatrix, TriangleType::upper> ? TriangleType::upper : TriangleType::lower))>
#endif
  struct TriangularAdapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_triangular_expr : std::false_type {};

      template<typename NestedMatrix, TriangleType triangle_type>
      struct is_triangular_expr<TriangularAdapter<NestedMatrix, triangle_type>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a triangular matrix based on the Eigen library (i.e., TriangularAdapter).
     */
    template<typename T>
#ifdef __cpp_concepts
    concept triangular_expr = detail::is_triangular_expr<std::decay_t<T>>::value;
#else
    constexpr bool triangular_expr = detail::is_triangular_expr<std::decay_t<T>>::value;
#endif
  }


  // ------------------------------------------ //
  //  VectorSpaceAdapter; vector_space_adapter  //
  // ------------------------------------------ //

  /**
   * \brief An adapter that adds vector space descriptors for each index.
   * \details Any vector space descriptors associated with NestedObject are overwritten.
   * \tparam Arg An \ref indexible object.
   * \taram Vs A set of \ref coordinate::pattern objects
   */
#ifdef __cpp_concepts
  template<indexible NestedObject, pattern_collection Descriptors>
  requires internal::not_more_fixed_than<NestedObject, Descriptors> and (not internal::less_fixed_than<NestedObject, Descriptors>) and
    internal::maybe_same_shape_as_vector_space_descriptors<NestedObject, Descriptors>
#else
  template<typename NestedObject, typename Descriptors>
#endif
  struct VectorSpaceAdapter;

  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_vector_space_adapter : std::false_type {};

      template<typename NestedObject, typename Descriptors>
      struct is_vector_space_adapter<VectorSpaceAdapter<NestedObject, Descriptors>> : std::true_type {};
    } // namespace detail


    /**
     * \internal
     * \brief Specifies that T is a VectorSpaceAdapter.
     */
    template<typename T>
#ifdef __cpp_concepts
    concept vector_space_adapter =
#else
    constexpr bool vector_space_adapter =
#endif
      detail::is_vector_space_adapter<std::decay_t<T>>::value;

  } // namespace internal


  // -------------------------------------------------------- //
  //  FromEuclideanExpr, from_euclidean_expr, euclidean_expr  //
  // -------------------------------------------------------- //

  /**
   * \brief An expression that transforms angular or other modular vector space descriptors back from Euclidean space.
   * \details This is the counterpart expression to ToEuclideanExpr.
   * \tparam NestedObject The pre-transformed column vector, or set of column vectors in the form of a matrix.
   * \tparam RowDescriptor The \ref coordinate::pattern of the first index.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject, coordinate::pattern RowDescriptor> requires
    compares_with<vector_space_descriptor_of<NestedObject, 0>, coordinate::Dimensions<coordinate::euclidean_size_of_v<RowDescriptor>>, equal_to<>, Applicability::permitted>
#else
  template<typename NestedMatrix, typename RowDescriptor>
#endif
  struct FromEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_from_euclidean_expr : std::false_type {};

    template<typename NestedMatrix, typename D>
    struct is_from_euclidean_expr<FromEuclideanExpr<NestedMatrix, D>> : std::true_type {};
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
   * \brief An expression that transforms vector space descriptors into Euclidean space for application of directional statistics.
   * \details This is the counterpart expression to FromEuclideanExpr.
   * \tparam NestedObject The pre-transformed column vector, or set of column vectors in the form of a matrix.
   */
#ifdef __cpp_concepts
  template<indexible NestedObject> requires (not from_euclidean_expr<NestedObject>)
#else
  template<typename NestedObject>
#endif
  struct ToEuclideanExpr;


  namespace detail
  {
    template<typename T>
    struct is_to_euclidean_expr : std::false_type {};

    template<typename NestedObject>
    struct is_to_euclidean_expr<ToEuclideanExpr<NestedObject>> : std::true_type {};
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


  // --------------------- //
  //  Deprecated adapters  //
  // --------------------- //

  /**
   * \brief A matrix with typed rows and columns.
   * \details It is a wrapper for a native matrix type from a supported matrix library such as Eigen.
   * The matrix can be thought of as a tests from X to Y, where the coefficients for each of X and Y are typed.
   * Example declarations:
   * - <code>Matrix<std::tuple<Axis, Axis, angle::Radians>, Dimensions<2>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * - <code>Matrix<double, std::tuple<Axis, Axis, angle::Radians>, Dimensions<2>,
   * eigen_matrix_t<double, 3, 2>> x;</code>
   * \tparam RowCoefficients A set of \ref OpenKalman::coefficients "coefficients" (e.g., Axis, Spherical, etc.)
   * corresponding to the rows.
   * \tparam ColumnCoefficients Another set of \ref OpenKalman::coefficients "coefficients" corresponding
   * to the columns.
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_pattern RowCoefficients, fixed_pattern ColumnCoefficients, typed_matrix_nestable NestedMatrix>
  requires (coordinate::size_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (coordinate::size_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and
    (dynamic_pattern<RowCoefficients> == dynamic_dimension<NestedMatrix, 0>) and
    (dynamic_pattern<ColumnCoefficients> == dynamic_dimension<NestedMatrix, 1>)
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
   * <code>Mean<std::tuple<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * \tparam Descriptor Coefficient types of the mean (e.g., Axis, Polar).
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_pattern Descriptor, typed_matrix_nestable NestedMatrix> requires
  (coordinate::size_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
  (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Descriptor, typename NestedMatrix>
#endif
  struct Mean;


  namespace internal
  {
    template<typename Descriptor, typename NestedMatrix>
    struct is_mean<Mean<Descriptor, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Similar to a Mean, but the coefficients are transformed into Euclidean space, based on their type.
   * \details Means containing angles should be converted to EuclideanMean before taking an average or weighted average.
   * Example declaration:
   * <code>EuclideanMean<std::tuple<Axis, Axis, angle::Radians>, 1, eigen_matrix_t<double, 4, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with the last two of the dimensions representing the angle::Radians coefficient
   * transformed to x and y locations on a unit circle associated with the angle::Radians-type coefficient.
   * \tparam Descriptor A set of coefficients (e.g., Axis, angle::Radians, Polar, etc.)
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<fixed_pattern Descriptor, typed_matrix_nestable NestedMatrix> requires
  (coordinate::euclidean_size_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Descriptor, typename NestedMatrix>
#endif
  struct EuclideanMean;


  namespace internal
  {
    template<typename Descriptor, typename NestedMatrix>
    struct is_euclidean_mean<EuclideanMean<Descriptor, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief A self-adjoint Covariance matrix.
   * \details The coefficient types for the rows are the same as for the columns.
   * \tparam Descriptor Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is triangular, the native matrix will be multiplied by its transpose
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<fixed_pattern Descriptor, covariance_nestable NestedMatrix> requires
    (coordinate::size_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and value::number<scalar_type_of_t<NestedMatrix>>
#else
  template<typename Descriptor, typename NestedMatrix>
#endif
  struct Covariance;


  namespace internal
  {
    template<typename Descriptor, typename NestedMatrix>
    struct is_self_adjoint_covariance<Covariance<Descriptor, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * \details If S is a SquareRootCovariance, S*transpose(S) is a Covariance.
   * If NestedMatrix is triangular, the SquareRootCovariance has the same triangle type (upper or lower). If NestedMatrix
   * is self-adjoint, the triangle type of SquareRootCovariance is considered either upper ''or'' lower.
   * \tparam Descriptor Coefficient types.
   * \tparam NestedMatrix The underlying native matrix or matrix expression. It can be either self-adjoint or
   * (either upper or lower) triangular. If it is self-adjoint, the native matrix will be Cholesky-factored
   * when converted to a Matrix or when used in mathematical expressions. The self-adjoint and triangular versions
   * are functionally identical, but often the triangular version is more efficient.
   */
#ifdef __cpp_concepts
  template<fixed_pattern Descriptor, covariance_nestable NestedMatrix> requires
    (coordinate::size_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and value::number<scalar_type_of_t<NestedMatrix>>
#else
  template<typename Descriptor, typename NestedMatrix>
#endif
  struct SquareRootCovariance;


  // ------------------- //
  //  Internal adapters  //
  // ------------------- //

  namespace internal
  {
    /**
     * \internal
     * \brief A wrapper for \ref indexible objects so that they are treated exactly as native objects within a library.
     * \tparam NestedObject An indexible object that may or may not be in a library of interest.
     * \tparam LibraryObject Any object from the library to which this wrapper is to be associated.
     * arguments necessary to construct the object, which will be stored internally.
     */
  #ifdef __cpp_concepts
    template<indexible NestedObject, indexible LibraryObject>
  #else
    template<typename NestedObject, typename LibraryObject>
  #endif
    struct LibraryWrapper;


    namespace detail
    {
      template<typename T>
      struct is_library_wrapper : std::false_type {};

      template<typename N, typename L>
      struct is_library_wrapper<internal::LibraryWrapper<N, L>> : std::true_type {};
    } // namespace detail


    /**
     * \internal
     * \brief T is a \ref internal::LibraryWrapper "LibraryWrapper".
     */
    template<typename T>
#ifdef __cpp_concepts
    concept library_wrapper =
#else
    constexpr bool library_wrapper =
#endif
      detail::is_library_wrapper<std::decay_t<T>>::value;


    /**
     * \internal
     * \brief Wraps a dynamic-sized input, immutably, in a wrapper that has one or more fixed dimensions.
     * \tparam NestedMatrix The underlying native matrix or matrix expression.
     * \tparam Descriptors A \ref pattern_tuple (preferably but not necessarily a \ref fixed_pattern_tuple).
     * If this set is empty, the object is treated as a \ref one_dimensional.
     */
  #ifdef __cpp_concepts
    template<indexible NestedObject, pattern_tuple Descriptors> requires
      compatible_with_vector_space_descriptor_collection<NestedObject, Descriptors> and
      internal::not_more_fixed_than<NestedObject, Descriptors> and internal::less_fixed_than<NestedObject, Descriptors>
  #else
    template<typename NestedObject, typename Descriptors>
  #endif
    struct FixedSizeAdapter;


  namespace detail
  {
    template<typename T>
    struct is_fixed_size_adapter : std::false_type {};

    template<typename NestedMatrix, typename Descriptors>
    struct is_fixed_size_adapter<FixedSizeAdapter<NestedMatrix, Descriptors>> : std::true_type {};
  }


  /**
   * \internal
   * \brief Specifies that T is a FixedSizeAdapter.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept fixed_size_adapter =
#else
  constexpr bool fixed_size_adapter =
#endif
    detail::is_fixed_size_adapter<std::decay_t<T>>::value;


    template<typename Descriptor, typename NestedMatrix>
    struct is_triangular_covariance<SquareRootCovariance<Descriptor, NestedMatrix>> : std::true_type {};

  } // namespace internal


  /**
   * \brief A Gaussian distribution, defined in terms of a Mean and a Covariance.
   * \tparam Descriptor Coefficient types.
   * \tparam MeanNestedMatrix The underlying native matrix for the Mean.
   * \tparam CovarianceNestedMatrix The underlying native matrix (triangular or self-adjoint) for the Covariance.
   * \tparam random_number_engine A random number engine compatible with the c++ standard library (e.g., std::mt19937).
   * \todo Change to std::mt19937_64 ?
   */
#ifdef __cpp_concepts
  template<
    fixed_pattern Descriptor,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine = std::mt19937> requires
      (index_dimension_of_v<MeanNestedMatrix, 0> == index_dimension_of_v<CovarianceNestedMatrix, 0>) and
      (index_dimension_of_v<MeanNestedMatrix, 1> == 1) and
      (std::is_same_v<scalar_type_of_t<MeanNestedMatrix>,
        scalar_type_of_t<CovarianceNestedMatrix>>)
#else
  template<
    typename Descriptor,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine = std::mt19937>
#endif
  struct GaussianDistribution;


  namespace internal
  {
    template<typename Descriptor, typename MeanNestedMatrix, typename CovarianceNestedMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<Descriptor, MeanNestedMatrix, CovarianceNestedMatrix, re>>
      : std::true_type {};
  }

} // OpenKalman

#endif //OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
