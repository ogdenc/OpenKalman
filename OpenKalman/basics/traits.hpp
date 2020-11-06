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

  // --------------------- //
  //    covariance_base    //
  // --------------------- //

  /// Tests whether an object is a native base matrix suitable for Covariance or SquareRootCovariance.
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename Enable = void>
#endif
  struct is_covariance_base : internal::class_trait<is_covariance_base, T> {};

  /// Helper template for is_covariance_base.
  template<typename T>
  inline constexpr bool is_covariance_base_v = is_covariance_base<T>::value;

#ifdef __cpp_concepts
  template<typename T>
  concept covariance_base = is_covariance_base_v<T>;
#else
  /// Helper template for is_covariance_base.
  template<typename T>
  inline constexpr bool covariance_base = is_covariance_base<T>::value;
#endif

  // ----------------------- //
  //    typed_matrix_base    //
  // ----------------------- //

  /// Whether an object is a base for a typed matrix.
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename Enable = void>
#endif
  struct is_typed_matrix_base : internal::class_trait<is_typed_matrix_base, T> {};

  /// Helper template for is_typed_matrix_base.
  template<typename T>
  inline constexpr bool is_typed_matrix_base_v = is_typed_matrix_base<T>::value;


#ifdef __cpp_concepts
  template<typename T>
  concept typed_matrix_base = is_typed_matrix_base_v<T>;
#else
  /// Helper template for is_typed_matrix_base.
  template<typename T>
  inline constexpr bool typed_matrix_base = is_typed_matrix_base<T>::value;
#endif


  /**
   * A matrix with typed rows and columns.
   *
   * The matrix can be thought of as a transformation from X to Y, where the coefficients for each of X and Y are typed.
   * Example declaration:
   * <code>Matrix<double, Coefficients<Axis, Axis, Angle>, Coefficients<Axis, Axis>> x;</code>
   * @tparam RowCoefficients A set of coefficients (e.g., Axis, Spherical, etc.) corresponding to the rows.
   * @tparam ColumnCoefficients Another set of coefficients corresponding to the columns.
   * @tparam ArgType The base matrix type.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_base ArgType> requires
    (RowCoefficients::size == MatrixTraits<ArgType>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<ArgType>::columns)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
#endif
  struct Matrix;


  /**
   * @brief A set of column vectors representing one or more means.
   * Generally, a column vector representing a mean. Alternatively, it can be a 2D matrix representing a collection of
   * column vectors of the same coefficient types, each column vector representing a distinct mean.
   * Example declaration:
   * <code>Mean<Coefficients<Axis, Axis, Angle>, 1, Eigen::Matrix<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an Angle, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * @tparam Coefficients Coefficient types of the mean (e.g., Axis, Polar).
   * @tparam BaseMatrix Regular matrix on which the mean is based (usually a column vector).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct Mean;


  /**
   * @brief The underlying class representing the Euclidean space version of a mean, with typed coefficients.
   *
   * Example declaration:
   * <code>EuclideanMean<Coefficients<Axis, Axis, Angle>, 1, Eigen::Matrix<double, 3, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an Angle, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with two of the dimensions representing the Angle coefficient.
   * @tparam Coefficients A set of coefficients (e.g., Angle, Polar, etc.)
   * @tparam BaseMatrix The mean's base type. This is a column vector or a matrix (considered as a collection of column vectors).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base BaseMatrix> requires
  (Coefficients::dimension == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct EuclideanMean;


  /**
   * A Covariance matrix.
   * @tparam Coefficients Coefficient types.
   * @tparam BaseMatrix Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct Covariance;


  /**
   * @brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * If S is a SquareRootCovariance, S*S.transpose() is a Covariance
   * @tparam Coefficients Coefficient types.
   * @tparam BaseMatrix Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct SquareRootCovariance;


  /**
   * @brief A Gaussian distribution, defined in terms of a mean vector and a covariance matrix.
   * @tparam Coefficients Coefficient types.
   * @tparam ArgMean Underlying type for Mean.
   * @tparam ArgMoment Underlying type for Moment.
   */
  template<
    typename Coefficients,
    typename MeanBase,
    typename CovarianceBase,
    typename random_number_engine>
  struct GaussianDistribution;


  // --------- //
  //   Means   //
  // --------- //


  namespace detail
  {
    template<typename T>
    struct is_mean : internal::class_trait<is_mean, T> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_mean<Mean<Coefficients, BaseMatrix>> : std::true_type {};
  }

  /** T is a mean (i.e., is a specialization of the class Mean).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept mean = detail::is_mean<T>::value;
#else
  inline constexpr bool mean = detail::is_mean<T>::value;
#endif


#ifdef __cpp_concepts
  /** T is a wrapped mean (i.e., its row coefficients have at least one type that requires wrapping).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept wrapped_mean = mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  namespace detail
  {
    template<typename T>
    struct is_wrapped_mean : internal::class_trait<is_wrapped_mean, T> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_wrapped_mean<Mean<Coefficients, BaseMatrix>> : std::bool_constant<not Coefficients::axes_only> {};
  }

  /// T is a wrapped mean (i.e., its row coefficients have at least one type that requires wrapping).
  template<typename T>
  inline constexpr bool wrapped_mean = detail::is_wrapped_mean<T>::value;
#endif


  // ------------------- //
  //   Euclidean means   //
  // ------------------- //

  namespace detail
  {
    template<typename T>
    struct is_euclidean_mean : internal::class_trait<is_euclidean_mean, T> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_euclidean_mean<EuclideanMean<Coefficients, BaseMatrix>> : std::true_type {};
  }

  /** T is a Euclidean mean (i.e., is a specialization of the class EuclideanMean).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_mean = detail::is_euclidean_mean<T>::value;
#else
  inline constexpr bool euclidean_mean = detail::is_euclidean_mean<T>::value;
#endif


#ifdef __cpp_concepts
  /** T is a euclidean_mean that actually has coefficients that are transformed to Euclidean space.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept euclidean_transformed = euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_euclidean_transformed : internal::class_trait<is_euclidean_transformed, T, Enable> {};

    template<typename T>
    struct is_euclidean_transformed<T, std::enable_if_t<euclidean_mean<T>>>
      : std::bool_constant<not MatrixTraits<T>::RowCoefficients::axes_only> {};
  }

  // T is a euclidean_mean that actually has coefficients that are transformed to Euclidean space.
  template<typename T>
  inline constexpr bool euclidean_transformed = detail::is_euclidean_transformed<T>::value;
#endif


  // ------------------ //
  //   typed matrices   //
  // ------------------ //

  namespace detail
  {
    template<typename T>
    struct is_matrix : internal::class_trait<is_matrix, T> {};

    template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
    struct is_matrix<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>> : std::true_type {};
  }

  /** T is a typed matrix (i.e., is a specialization of Matrix, Mean, or EuclideanMean).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix = mean<T> or euclidean_mean<T> or detail::is_matrix<T>::value;
#else
  inline constexpr bool typed_matrix = mean<T> or euclidean_mean<T> or detail::is_matrix<T>::value;
#endif


#ifdef __cpp_concepts
  /** T is a column vector or set of column vectors (i.e., the columns all have type Axis).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept column_vector = typed_matrix<T> and MatrixTraits<T>::ColumnCoefficients::axes_only;
#else
  namespace detail
  {
    template<typename T>
    struct is_column_vector : internal::class_trait<is_column_vector, T> {};

    template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
    struct is_column_vector<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>>
      : std::bool_constant<ColumnCoefficients::axes_only> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_column_vector<Mean<Coefficients, BaseMatrix>> : std::true_type {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_column_vector<EuclideanMean<Coefficients, BaseMatrix>> : std::true_type {};
  }

  /// T is a column vector or set of column vectors (i.e., the columns all have type Axis).
  template<typename T>
  inline constexpr bool column_vector = detail::is_column_vector<T>::value;
#endif


  // ------------------------------------ //
  //  square root (Cholesky) covariances  //
  // ------------------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_square_root_covariance : internal::class_trait<is_square_root_covariance, T> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_square_root_covariance<SquareRootCovariance<Coefficients, BaseMatrix>> : std::true_type {};
  }

  /** T is a square root (Cholesky) covariance matrix with typed rows and columns. The rows and columns have the same type.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept square_root_covariance = detail::is_square_root_covariance<T>::value;
#else
  inline constexpr bool square_root_covariance = detail::is_square_root_covariance<T>::value;
#endif


  // ------------------------ //
  //  covariances in general  //
  // ------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_non_Cholesky_covariance : internal::class_trait<is_non_Cholesky_covariance, T> {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_non_Cholesky_covariance<Covariance<Coefficients, BaseMatrix>> : std::true_type {};
  }

  /** T is a covariance matrix of any kind, including a square_root_covariance. The rows and columns have the same type.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance = square_root_covariance<T> or detail::is_non_Cholesky_covariance<T>::value;
#else
  inline constexpr bool covariance = square_root_covariance<T> or detail::is_non_Cholesky_covariance<T>::value;
#endif


  // --------------- //
  //  distributions  //
  // --------------- //

  namespace detail
  {
    template<typename T>
    struct is_gaussian_distribution : internal::class_trait<is_gaussian_distribution, T> {};

    template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
      : std::true_type {};
  }

  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = detail::is_gaussian_distribution<T>::value;
#else
  inline constexpr bool gaussian_distribution = detail::is_gaussian_distribution<T>::value;
#endif


  /** T is a distribution.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  inline constexpr bool distribution = gaussian_distribution<T>;
#endif


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


  // ------------ //
  //  Strictness  //
  // ------------ //

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
    struct strict_impl<T, std::enable_if_t<distribution<T>>>
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


  // -------------- //
  //  Other traits  //
  // -------------- //

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
