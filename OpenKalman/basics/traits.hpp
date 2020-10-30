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
 * @file Traits.h
 * A header file containing forward declarations for all OpenKalman traits.
 */

#ifndef OPENKALMAN_TRAITS_HPP
#define OPENKALMAN_TRAITS_HPP

#include <type_traits>

namespace OpenKalman
{
  // ---------------- //
  //   Coefficients   //
  // ---------------- //

  namespace internal
  {
    /*
     * A type trait testing whether T is an atomic group of coefficients.
     *
     * The atomic coefficient groups are the following:
     * - Axis
     * - Circle (or alias Angle)
     * - Distance
     * - Inclination
     * - Polar
     * - Spherical
     * Atomic coefficient groups may be combined into composite coefficient sets by passing them as template
     * arguments to Coefficients. For example Coefficients<Axis, Polar<Distance, Angle>> is a set comprising an axis and
     * a set of polar coordinates.
     */
    template<typename T>
    struct is_atomic_coefficient_group;

    /*
     * A type trait testing whether T is a composite set of coefficient groups.
     *
     * This corresponds to any specialization of the class Coefficients. Composite coefficients can, themselves,
     * comprise groups of other composite components. For example, Coefficients<Axis, Coefficients<Axis, Angle>>
     * tests positive for is_composite_coefficients.
     */
    template<typename T>
    struct is_composite_coefficients;
  }

#ifdef __cpp_concepts
  /**
   * T is a coefficient group.
   *
   * A coefficient group may consist of some combination of any of the following:
   * - Axis
   * - Circle (including alias Angle)
   * - Distance
   * - Inclination
   * - Polar
   * - Spherical
   * - Coefficient (a composite coefficient including any other coefficient group).
   * Examples: Axis, Angle, Coefficient<Axis, Axis>, Coefficient<Angle, Coefficient<Axis, Angle>>.
   */
  template<typename T>
  concept coefficients =
    internal::is_composite_coefficients<T>::value or
    internal::is_atomic_coefficient_group<T>::value;
#else
  /// A type trait testing whether T is a (composite or atomic) coefficient group.
  template<typename T>
  struct is_coefficients : std::integral_constant<bool,
    internal::is_composite_coefficients<T>::value or internal::is_atomic_coefficient_group<T>::value> {};

  /// Helper template for is_coefficients.
  template<typename T>
  inline constexpr bool is_coefficients_v = is_coefficients<T>::value;
#endif


  /**
   * Tests whether two sets of coefficients are equivalent to each other.
   *
   * For example, <code>is_equivalent<Axis, Coefficients<Axis>></code> tests true.
   */
#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U, typename Enable = void>
#endif
  struct is_equivalent;


  /// Helper template for is_equivalent.
#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U>
#endif
  inline constexpr bool is_equivalent_v = is_equivalent<T, U>::value;

  /**
   * T is equivalent to U
   * For example, <code>equivalent<Axis, Coefficients<Axis>></code> returns <code>true</code>.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
  concept equivalent = is_equivalent_v<T, U>;
#endif


  /**
   * Tests whether one set of coefficients is a pre-fix for another set.
   *
   * For example, <code>is_prefix<Coefficients<Axis>, Coefficients<Axis, Angle>></code> tests true.
   */
#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U, typename Enable = void>
#endif
  struct is_prefix;


  /// Helper template for is_prefix.
#ifdef __cpp_concepts
  template<coefficients T, coefficients U>
#else
  template<typename T, typename U>
#endif
  inline constexpr bool is_prefix_v = is_prefix<T, U>::value;


  /**
   * T is a prefix of U
   * For example, <code>prefix<Coefficients<Axis>, Coefficients<Axis, Angle>></code> returns <code>true</code>.
   */
#ifdef __cpp_concepts
  template<typename T, typename U>
  concept prefix = is_prefix_v<T, U>;
#endif


  // ------------------ //
  //   General traits   //
  // ------------------ //

  /**
   * Describes the traits of a matrix, such as its dimensions, coefficient types, etc.
   * @addtogroup Traits
   * @tparam M The matrix type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename M>
  struct MatrixTraits {};

  template<typename M> requires std::is_reference_v<M> or std::is_const_v<std::remove_reference_t<M>>
  struct MatrixTraits<M> : public MatrixTraits<std::decay_t<M>> {};
#else
  template<typename M, typename Enable = void> /// @tparam Enable A dummy variable to enable the class.
  struct MatrixTraits {};

  template<typename M>
  struct MatrixTraits<M&> : MatrixTraits<M> {};

  template<typename M>
  struct MatrixTraits<M&&> : MatrixTraits<M> {};

  template<typename M>
  struct MatrixTraits<const M> : MatrixTraits<M> {};
#endif


  namespace internal
  {
    /*
     * A class trait that also applies to ref and cv-qualified classes.
     * @tparam Trait The trait, a template template parameter that takes T and Es... as parameters.
     */
    template<template<typename T, typename...Es> typename Trait, typename T, typename...Es>
    struct class_trait : std::false_type {};

    template<template<typename T, typename...Es> typename Trait, typename T, typename...Es>
    struct class_trait<Trait, T&, Es...> : Trait<T, Es...> {};

    template<template<typename T, typename...Es> typename Trait, typename T, typename...Es>
    struct class_trait<Trait, T&&, Es...> : Trait<T, Es...> {};

    template<template<typename T, typename...Es> typename Trait, typename T, typename...Es>
    struct class_trait<Trait, const T, Es...> : Trait<T, Es...> {};
  }


  // ---------------------------- //
  //  Traits for square matrices  //
  // ---------------------------- //

  /// Whether an object is a Covariance or SquareRootCovariance.
  template<typename T>
  struct is_covariance : internal::class_trait<is_covariance, T> {};

  /// Helper template for is_covariance.
  template<typename T>
  inline constexpr bool is_covariance_v = is_covariance<T>::value;

  /// Whether an object is a Cholesky square root (e.g., SquareRootCovariance).
  template<typename T>
  struct is_square_root : internal::class_trait<is_square_root, T> {};

  /// Helper template for is_square_root.
  template<typename T>
  inline constexpr bool is_square_root_v = is_square_root<T>::value;

  /// Whether an object is a diagonal matrix.
  template<typename T, typename Enable = void>
  struct is_zero : internal::class_trait<is_zero, T, Enable> {};

  /// Helper template for is_zero.
  template<typename T>
  inline constexpr bool is_zero_v = is_zero<T>::value;

  /// Whether an object is an identity matrix.
  template<typename T, typename Enable = void>
  struct is_identity : internal::class_trait<is_identity, T, Enable> {};

  /// Helper template for is_identity.
  template<typename T>
  inline constexpr bool is_identity_v = is_identity<T>::value;

  /// Whether an object is a 1-by-1 matrix.
  template<typename T, typename Enable = void>
  struct is_1by1 : internal::class_trait<is_1by1, T, Enable> {};

  /// Defining a 1-by-1 matrix.
  template<typename T>
  struct is_1by1<T, std::enable_if_t<MatrixTraits<T>::dimension == 1 and MatrixTraits<T>::columns == 1>>
    : std::true_type {};

  /// Helper template for is_1by1.
  template<typename T>
  inline constexpr bool is_1by1_v = is_1by1<T>::value;

  /// Whether an object is a diagonal matrix.
  template<typename T, typename Enable = void>
  struct is_diagonal : internal::class_trait<is_diagonal, T, Enable> {};

  /// Zero and identity matrices are diagonal.
  template<typename T>
  struct is_diagonal<T, std::enable_if_t<is_zero_v<T> or is_identity_v<T> or is_1by1_v<T>>> : std::true_type {};

  /// Helper template for is_diagonal.
  template<typename T>
  inline constexpr bool is_diagonal_v = is_diagonal<T>::value;

  /// Whether a covariance is in the form of a Cholesky decomposition.
  template<typename T, typename Enable = void>
  struct is_Cholesky : internal::class_trait<is_Cholesky, T,  Enable> {};

  /// Helper template for is_Cholesky.
  template<typename T>
  inline constexpr bool is_Cholesky_v = is_Cholesky<T>::value;

  /// Whether an object is a self-adjoint matrix.
  template<typename T, typename Enable = void>
  struct is_self_adjoint : internal::class_trait<is_self_adjoint, T, Enable> {};

  /// Helper template for is_self_adjoint.
  template<typename T>
  inline constexpr bool is_self_adjoint_v = is_self_adjoint<T>::value;

  /// Diagonal matrices are self-adjoint.
  template<typename T>
  struct is_self_adjoint<T, std::enable_if_t<is_diagonal_v<T>>> : std::true_type {};

  /// Whether an object is a lower triangular matrix.
  template<typename T, typename Enable = void>
  struct is_lower_triangular : internal::class_trait<is_lower_triangular, T, Enable> {};

  /// Diagonal matrices are lower-triangular.
  template<typename T>
  struct is_lower_triangular<T, std::enable_if_t<is_diagonal_v<T>>> : std::true_type {};

  /// Helper template for is_lower_triangular.
  template<typename T>
  inline constexpr bool is_lower_triangular_v = is_lower_triangular<T>::value;

  /// Whether an object is an upper triangular matrix.
  template<typename T, typename Enable = void>
  struct is_upper_triangular : internal::class_trait<is_upper_triangular, T, Enable> {};

  /// Diagonal matrices are upper-triangular.
  template<typename T>
  struct is_upper_triangular<T, std::enable_if_t<is_diagonal_v<T>>> : std::true_type {};

  /// Whether an object is a triangular matrix.
  /// Helper template for is_upper_triangular.
  template<typename T>
  inline constexpr bool is_upper_triangular_v = is_upper_triangular<T>::value;

  template<typename T>
  struct is_triangular : std::bool_constant<is_lower_triangular_v<T> or is_upper_triangular_v<T>> {};

  /// Helper template for is_triangular.
  template<typename T>
  inline constexpr bool is_triangular_v = is_triangular<T>::value;


  enum struct TriangleType { lower, upper, diagonal };

  /// Derive TriangleType from type traits.
  template<typename T, typename Enable = void>
  struct triangle_type_of : std::integral_constant<TriangleType, TriangleType::lower> {};

  template<typename T>
  struct triangle_type_of<T, std::enable_if_t<is_lower_triangular_v<T> and not is_diagonal_v<T>>> :
    std::integral_constant<TriangleType, TriangleType::lower> {};

  template<typename T>
  struct triangle_type_of<T, std::enable_if_t<is_upper_triangular_v<T> and not is_diagonal_v<T>>> :
    std::integral_constant<TriangleType, TriangleType::upper> {};

  template<typename T>
  struct triangle_type_of<T, std::enable_if_t<is_diagonal_v<T>>> :
    std::integral_constant<TriangleType, TriangleType::diagonal> {};

  /// Helper template for is_triangular.
  template<typename T>
  inline constexpr TriangleType triangle_type_of_v = triangle_type_of<T>::value;

  /// Whether an object is a base for Covariance or SquareRootCovariance.
  template<typename T, typename Enable = void>
  struct is_covariance_base : internal::class_trait<is_covariance_base, T> {};

  /// Helper template for is_covariance_base.
  template<typename T>
  inline constexpr bool is_covariance_base_v = is_covariance_base<T>::value;


  /////////////////////////////////
  //  Traits for typed matrices  //
  /////////////////////////////////

  /// Whether an object is a typed matrix (i.e., Mean, EuclideanMean, or Matrix).
  template<typename T>
  struct is_typed_matrix : internal::class_trait<is_typed_matrix, T> {};

  /// Helper template for is_covariance_base.
  template<typename T>
  inline constexpr bool is_typed_matrix_v = is_typed_matrix<T>::value;

  /// Whether an object is a mean.
  template<typename T>
  struct is_mean : internal::class_trait<is_mean, T> {};

  /// Helper template for is_mean.
  template<typename T>
  inline constexpr bool is_mean_v = is_mean<T>::value;

  /// Whether an object is a euclidean mean.
  template<typename T>
  struct is_Euclidean_mean : internal::class_trait<is_Euclidean_mean, T> {};

  /// Helper template for is_Euclidean_mean.
  template<typename T>
  inline constexpr bool is_Euclidean_mean_v = is_Euclidean_mean<T>::value;

  /// Whether an object is Euclidean-transformed.
  template<typename T, typename Enable = void>
  struct is_Euclidean_transformed : internal::class_trait<is_Euclidean_transformed, T, Enable> {};

  /// A EuclideanMean is Euclidean-transformed unless the coefficients are Axes only.
  template<typename T>
  struct is_Euclidean_transformed<T, std::enable_if_t<is_Euclidean_mean_v<T>>>
    : std::bool_constant<not MatrixTraits<T>::RowCoefficients::axes_only> {};

  /// Helper template for is_covariance_base.
  template<typename T>
  inline constexpr bool is_Euclidean_transformed_v = is_Euclidean_transformed<T>::value;

  /// Whether the matrix is wrapped.
  template<typename T, typename Enable = void>
  struct is_wrapped : internal::class_trait<is_wrapped, T, Enable> {};

  /// A Mean is wrapped unless the coefficients are Axes only.
  template<typename T>
  struct is_wrapped<T, std::enable_if_t<is_mean_v<T>>>
    : std::bool_constant<not MatrixTraits<T>::RowCoefficients::axes_only> {};

  /// Helper template for is_wrapped.
  template<typename T>
  inline constexpr bool is_wrapped_v = is_wrapped<T>::value;

  /// Whether the matrix is a column vector or set of column vectors.
  template<typename T, typename Enable = void>
  struct is_column_vector : internal::class_trait<is_column_vector, T,  Enable> {};

  /// A typed matrix is a column vector if the columns are Axes only.
  template<typename T>
  struct is_column_vector<T, std::enable_if_t<is_typed_matrix_v<T>>>
    : std::bool_constant<MatrixTraits<T>::ColumnCoefficients::axes_only> {};

  /// Helper template for is_column_vector.
  template<typename T>
  inline constexpr bool is_column_vector_v = is_column_vector<T>::value;


  /// Whether an object is a base for a typed matrix.
  template<typename T, typename Enable = void>
  struct is_typed_matrix_base : internal::class_trait<is_typed_matrix_base, T> {};

  /// Helper template for is_typed_matrix_base.
  template<typename T>
  inline constexpr bool is_typed_matrix_base_v = is_typed_matrix_base<T>::value;


  /////////////////////
  //  Distributions  //
  /////////////////////

  /// Whether an object is a distribution.
  template<typename T, typename Enable = void>
  struct is_distribution : internal::class_trait<is_distribution, T, Enable> {};

  /// Helper template for is_distribution.
  template<typename T>
  inline constexpr bool is_distribution_v = is_distribution<T>::value;


  /// Whether an object is a Gaussian distribution.
  template<typename T>
  struct is_Gaussian_distribution : internal::class_trait<is_Gaussian_distribution, T> {};

  /// Helper template for is_Gaussian_distribution.
  template<typename T>
  inline constexpr bool is_Gaussian_distribution_v = is_Gaussian_distribution<T>::value;


  template<typename T>
  struct is_distribution<T, std::enable_if_t<is_Gaussian_distribution_v<T>>>
    : std::true_type {};


  /**
   * @brief Traits of a distribution.
   * @tparam Dist Distribution.
   */
  template<typename Dist, typename T = void>
  struct DistributionTraits {};

  template<typename D, typename T>
  struct DistributionTraits<D&, T> : DistributionTraits<D, T> {};

  template<typename D, typename T>
  struct DistributionTraits<D&&, T> : DistributionTraits<D, T> {};

  template<typename D, typename T>
  struct DistributionTraits<const D, T> : DistributionTraits<D, T> {};


  ////////////////////
  //  Other traits  //
  ////////////////////

  /// Whether an expression is strict.
  template<typename T, typename Enable = void>
  struct is_strict : internal::class_trait<is_strict, T> {};

  /// Helper template for is_strict.
  template<typename T>
  inline constexpr bool is_strict_v = is_strict<T>::value;

  namespace detail
  {
    template<typename T, typename Enable = void>
    struct strict_impl { using type = typename MatrixTraits<T>::Strict; };

    template<typename T>
    struct strict_impl<T, std::enable_if_t<is_distribution_v<T>>>
    {
      using type = typename DistributionTraits<T>::Strict;
    };
  }

  template<typename T>
  using strict_t = std::conditional_t<is_strict_v<T>, std::decay_t<T>, typename detail::strict_impl<T>::type>;

  template<typename T>
  using lvalue_or_strict_t = std::conditional_t<std::is_lvalue_reference_v<T>, std::decay_t<T>, strict_t<T>>;

  /// Whether an expression is a strict, regular matrix.
  template<typename T, typename Enable = void>
  struct is_strict_matrix : internal::class_trait<is_strict_matrix, T> {};

  /// Helper template for is_strict.
  template<typename T>
  inline constexpr bool is_strict_matrix_v = is_strict_matrix<T>::value;

  template<typename T, std::size_t rows = MatrixTraits<T>::dimension, std::size_t cols = MatrixTraits<T>::columns>
  using strict_matrix_t = typename MatrixTraits<T>::template StrictMatrix<rows, cols>;


  /// Whether an object has elements that can be retrieved with N indices.
  template<typename T, std::size_t N, typename Enable = void>
  struct is_element_gettable : std::false_type {};

  template<typename T, std::size_t N, typename Enable>
  struct is_element_gettable<T&, N, Enable> : is_element_gettable<T, N, Enable> {};

  template<typename T, std::size_t N, typename Enable>
  struct is_element_gettable<T&&, N, Enable> : is_element_gettable<T, N, Enable> {};

  template<typename T, std::size_t N, typename Enable>
  struct is_element_gettable<const T, N, Enable> : is_element_gettable<T, N, Enable> {};

  /// Helper template for is_element_gettable.
  template<typename T, std::size_t N>
  inline constexpr bool is_element_gettable_v = is_element_gettable<T, N>::value;


  /// Whether an object has elements that can be set with N indices.
  template<typename T, std::size_t N, typename Enable = void>
  struct is_element_settable : std::false_type {};

  template<typename T, std::size_t N, typename Enable>
  struct is_element_settable<T&, N, Enable> : is_element_settable<T, N, Enable> {};

  template<typename T, std::size_t N, typename Enable>
  struct is_element_settable<T&&, N, Enable> : is_element_settable<T, N, Enable> {};

  /// Helper template for is_element_settable.
  template<typename T, std::size_t N>
  inline constexpr bool is_element_settable_v = is_element_settable<T, N>::value;

}

#endif //OPENKALMAN_TRAITS_HPP