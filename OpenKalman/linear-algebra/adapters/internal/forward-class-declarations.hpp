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

#include "patterns/patterns.hpp"

namespace OpenKalman
{
  // ------------------------------------------ //
  //  to_diagonal_adapter, internal::diagonal_expr  //
  // ------------------------------------------ //

  /**
   * \brief An adapter for creating a diagonal matrix or tensor.
   * \details The matrix is guaranteed to be diagonal.
   * Implicit conversions are available from any \ref diagonal_matrix of compatible size.
   * \tparam NestedMatrix A column vector expression defining the diagonal elements.
   * object_traits outside the diagonal are automatically 0.
   */
#ifdef __cpp_concepts
  template<vector<0, applicability::permitted> NestedMatrix>
#else
  template<typename NestedMatrix>
#endif
  struct to_diagonal_adapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_diagonal_expr : std::false_type {};

      template<typename NestedMatrix>
      struct is_diagonal_expr<to_diagonal_adapter<NestedMatrix>> : std::true_type {};
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
  }


  // -------------------------------------------- //
  //  hermitian_adapter, internal::hermitian_expr  //
  // -------------------------------------------- //

#ifdef __cpp_concepts
  template<square_shaped<applicability::permitted> NestedMatrix, triangle_type storage_triangle =
      triangular_matrix<NestedMatrix, triangle_type::lower> ? triangle_type::lower : triangle_type::upper> requires
    (index_count_v<NestedMatrix> <= 2) and
    (storage_triangle == triangle_type::lower or storage_triangle == triangle_type::upper) and
    (not constant_matrix<NestedMatrix> or values::not_complex<constant_value<NestedMatrix>>) and
    (not constant_diagonal_matrix<NestedMatrix> or values::not_complex<constant_diagonal_value<NestedMatrix>>)
#else
  template<typename NestedMatrix, triangle_type storage_triangle =
    triangular_matrix<NestedMatrix, triangle_type::lower> ? triangle_type::lower : triangle_type::upper>
#endif
  struct hermitian_adapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_hermitian_expr : std::false_type {};

      template<typename NestedMatrix, triangle_type storage_triangle>
      struct is_hermitian_expr<hermitian_adapter<NestedMatrix, storage_triangle>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a self-adjoint matrix based on the Eigen library (i.e., hermitian_adapter).
     */
    template<typename T>
#ifdef __cpp_concepts
    concept hermitian_expr = detail::is_hermitian_expr<std::decay_t<T>>::value;
#else
    constexpr bool hermitian_expr = detail::is_hermitian_expr<std::decay_t<T>>::value;
#endif
  }


  // ---------------------------------------------- //
  //  triangular_adapter, internal::triangular_expr  //
  // ---------------------------------------------- //

#ifdef __cpp_concepts
  template<
    square_shaped<applicability::permitted> NestedMatrix,
    triangle_type tri = (diagonal_matrix<NestedMatrix> ? triangle_type::diagonal :
      (triangular_matrix<NestedMatrix, triangle_type::upper> ? triangle_type::upper : triangle_type::lower))>
    requires (index_count_v<NestedMatrix> <= 2)
#else
  template<typename NestedMatrix, triangle_type tri = (diagonal_matrix<NestedMatrix> ? triangle_type::diagonal :
    (triangular_matrix<NestedMatrix, triangle_type::upper> ? triangle_type::upper : triangle_type::lower))>
#endif
  struct triangular_adapter;


  namespace internal
  {
    namespace detail
    {
      template<typename T>
      struct is_triangular_expr : std::false_type {};

      template<typename NestedMatrix, triangle_type tri>
      struct is_triangular_expr<triangular_adapter<NestedMatrix, tri>> : std::true_type {};
    }


    /**
     * \brief Specifies that T is a triangular matrix based on the Eigen library (i.e., triangular_adapter).
     */
    template<typename T>
#ifdef __cpp_concepts
    concept triangular_expr = detail::is_triangular_expr<std::decay_t<T>>::value;
#else
    constexpr bool triangular_expr = detail::is_triangular_expr<std::decay_t<T>>::value;
#endif
  }


  // -------------------------------------------------------- //
  //  from_stat_space_adapter, from_stat_space_expr, euclidean_expr  //
  // -------------------------------------------------------- //

#ifdef __cpp_concepts
  template<indexible NestedObject, patterns::pattern RowDescriptor> requires
    compares_with<vector_space_descriptor_of<NestedObject, 0>, patterns::Dimensions<patterns::stat_dimension_of_v<RowDescriptor>>, equal_to<>, applicability::permitted>
#else
  template<typename NestedMatrix, typename RowDescriptor>
#endif
  struct from_stat_space_adapter;


  namespace detail
  {
    template<typename T>
    struct is_from_stat_space_expr : std::false_type {};

    template<typename NestedMatrix, typename D>
    struct is_from_stat_space_expr<from_stat_space_adapter<NestedMatrix, D>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients from Euclidean space (i.e., from_stat_space_adapter).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept from_stat_space_expr = detail::is_from_stat_space_expr<std::decay_t<T>>::value;
#else
  constexpr bool from_stat_space_expr = detail::is_from_stat_space_expr<std::decay_t<T>>::value;
#endif


  // ------------------------------------ //
  //  to_stat_space_adapter, to_euclidean_expr  //
  // ------------------------------------ //

#ifdef __cpp_concepts
  template<indexible Nested> requires (not from_stat_space_expr<Nested>)
#else
  template<typename Nested>
#endif
  struct to_stat_space_adapter;


  namespace detail
  {
    template<typename T>
    struct is_to_euclidean_expr : std::false_type {};

    template<typename NestedObject>
    struct is_to_euclidean_expr<to_stat_space_adapter<NestedObject>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is an expression converting coefficients to Euclidean space (i.e., to_stat_space_adapter).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#else
  constexpr bool to_euclidean_expr = detail::is_to_euclidean_expr<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is either \ref to_euclidean_expr or \ref from_stat_space_expr.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_expr = to_euclidean_expr<T> or from_stat_space_expr<T>;
#else
  constexpr bool euclidean_expr = from_stat_space_expr<T> or to_euclidean_expr<T>;
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
  requires (patterns::dimension_of_v<RowCoefficients> == index_dimension_of_v<NestedMatrix, 0>) and
    (patterns::dimension_of_v<ColumnCoefficients> == index_dimension_of_v<NestedMatrix, 1>) and
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
  (patterns::dimension_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
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
  (patterns::stat_dimension_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and (not std::is_rvalue_reference_v<NestedMatrix>)
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
    (patterns::dimension_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and values::number<scalar_type_of_t<NestedMatrix>>
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
    (patterns::dimension_of_v<Descriptor> == index_dimension_of_v<NestedMatrix, 0>) and
    (not std::is_rvalue_reference_v<NestedMatrix>) and values::number<scalar_type_of_t<NestedMatrix>>
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
    }


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
     * \tparam Descriptors A \ref pattern_collection.
     * If this set is empty, the object is treated as a \ref one_dimensional.
     */
  #ifdef __cpp_concepts
    template<indexible NestedObject, pattern_collection Descriptors> requires
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

  }


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

#endif
