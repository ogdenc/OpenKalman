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


   namespace internal
   {
    // Type trait testing whether coefficients T are equivalent to coefficients U.
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_equivalent_to;
   }


  /**
   * T is equivalent to U.
   *
   * For example, <code>equivalent_to<Axis, Coefficients<Axis>></code> returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept equivalent_to = internal::is_equivalent_to<T, U>::value;
#else
  inline constexpr bool equivalent_to = internal::is_equivalent_to<T, U>::value;
#endif


  namespace internal
  {
    // Type trait testing whether T (a set of coefficients) is a prefix of U.
#ifdef __cpp_concepts
    template<coefficients T, coefficients U>
#else
    template<typename T, typename U, typename Enable = void>
#endif
    struct is_prefix_of;
  }


  /**
   * T is a prefix of U
   *
   * For example, <code>prefix_of<Coefficients<Axis>, Coefficients<Axis, Angle>></code> returns <code>true</code>.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept prefix_of = internal::is_prefix_of<T, U>::value;
#else
  inline constexpr bool prefix_of = internal::is_prefix_of<T, U>::value;
#endif


  // ------------------ //
  //   General traits   //
  // ------------------ //

  /**
   * Describes the traits of a matrix T, such as its dimensions, coefficient types, etc.
   * @addtogroup Traits
   * @tparam T The matrix type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
  struct MatrixTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct MatrixTraits<T> : public MatrixTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< @tparam Enable A dummy variable to enable the class.
  struct MatrixTraits {};

  template<typename T>
  struct MatrixTraits<T&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<T&&> : MatrixTraits<T> {};

  template<typename T>
  struct MatrixTraits<const T> : MatrixTraits<T> {};
#endif


  /**
   * Describes the traits of a distribution type.
   * @addtogroup Traits
   * @tparam T The distribution type. The type is treated as non-qualified, even if it is const or a reference.
   */
#ifdef __cpp_concepts
  template<typename T>
  struct DistributionTraits {};

  template<typename T> requires std::is_reference_v<T> or std::is_const_v<std::remove_reference_t<T>>
  struct DistributionTraits<T> : DistributionTraits<std::decay_t<T>> {};
#else
  template<typename T, typename Enable = void> ///< @tparam Enable A dummy variable to enable the class.
  struct DistributionTraits {};

  template<typename T>
  struct DistributionTraits<T&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<T&&> : DistributionTraits<T> {};

  template<typename T>
  struct DistributionTraits<const T> : DistributionTraits<T> {};
#endif


  namespace internal
  {
    /*
     * A class trait that also applies to ref and cv-qualified classes.
     * @tparam Trait The trait, a template template parameter that takes T and Es... as parameters.
     */
#ifdef __cpp_concepts
    template<template<typename T> typename Trait, typename T>
    struct class_trait : std::false_type {};

    template<template<typename T> typename Trait, typename T>
    struct class_trait<Trait, T&> : Trait<T> {};

    template<template<typename T> typename Trait, typename T>
    struct class_trait<Trait, T&&> : Trait<T> {};

    template<template<typename T> typename Trait, typename T>
    struct class_trait<Trait, const T> : Trait<T> {};
#else
    template<template<typename T, typename...Enable> typename Trait, typename T, typename...Enable>
    struct class_trait : std::false_type {};

    template<template<typename T, typename...Enable> typename Trait, typename T, typename...Enable>
    struct class_trait<Trait, T&, Enable...> : Trait<T, Enable...> {};

    template<template<typename T, typename...Enable> typename Trait, typename T, typename...Enable>
    struct class_trait<Trait, T&&, Enable...> : Trait<T, Enable...> {};

    template<template<typename T, typename...Enable> typename Trait, typename T, typename...Enable>
    struct class_trait<Trait, const T, Enable...> : Trait<T, Enable...> {};
#endif

  } // namespace internal


  // --------------------- //
  //    covariance_base    //
  // --------------------- //

  namespace internal
  {
    // All true instances of is_covariance_base need to defined in each matrix interface.
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_covariance_base : class_trait<is_covariance_base, T> {};
  }

  /**
   * T is an acceptable base matrix for a covariance (including square_root_covariance).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance_base = internal::is_covariance_base<T>::value;
#else
  inline constexpr bool covariance_base = internal::is_covariance_base<T>::value;
#endif


  // ----------------------- //
  //    typed_matrix_base    //
  // ----------------------- //

  namespace internal
  {
    // All true instances of is_typed_matrix_base need to be defined in each matrix interface.
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct is_typed_matrix_base : class_trait<is_typed_matrix_base, T> {};
  }

  /**
   * T is an acceptable base for a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_base = internal::is_typed_matrix_base<T>::value;
#else
  inline constexpr bool typed_matrix_base = internal::is_typed_matrix_base<T>::value;
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
   * @details If S is a SquareRootCovariance, S*S.transpose() is a Covariance.
   * If BaseMatrix is triangular, the SquareRootCovariance has the same triangle type (upper or lower). If BaseMatrix
   * is self-adjoint, the triangle type of SquareRootCovariance is considered either upper ''or'' lower.
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

  /**
   * T is a Gaussian distribution.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = detail::is_gaussian_distribution<T>::value;
#else
  inline constexpr bool gaussian_distribution = detail::is_gaussian_distribution<T>::value;
#endif


  /** T is a statistical distribution of any kind that is defined in OpenKalman.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  inline constexpr bool distribution = gaussian_distribution<T>;
#endif


  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T>
    struct is_self_contained : internal::class_trait<is_self_contained, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_self_contained : internal::class_trait<is_self_contained, T, Enable> {};
#endif
  }

  /**
   * T is a self-contained matrix or expression (i.e., it can be the return value of a function).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_contained = internal::is_self_contained<T>::value;
#else
  inline constexpr bool self_contained = internal::is_self_contained<T>::value;
#endif


  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename Enable = void>
#endif
    struct self_contained_impl { using type = typename MatrixTraits<T>::SelfContained; };

#ifdef __cpp_concepts
    template<distribution T>
    struct self_contained_impl<T>
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<distribution<T>>>
#endif
    {
      using type = typename DistributionTraits<T>::SelfContained;
    };
  }


  /**
   * A self-contained version of type T.
   */
  template<typename T>
  using self_contained_t = std::conditional_t<
    self_contained<T>, std::decay_t<T>, typename detail::self_contained_impl<T>::type>;


  /**
   * A self-contained, native-matrix version of type T.
   * @tparam rows Number of rows in the native matrix (defaults to number of rows in T).
   * @tparam cols Number of columns in the native matrix (default to number of columns in T).
   */
  template<typename T, std::size_t rows = MatrixTraits<T>::dimension, std::size_t cols = MatrixTraits<T>::columns>
  using native_matrix_t = typename MatrixTraits<T>::template NativeMatrix<rows, cols>;


  /**
   *
   */
  template<typename T>
  using lvalue_or_self_contained_t = std::conditional_t<std::is_lvalue_reference_v<T>, std::decay_t<T>, self_contained_t<T>>;

  template<typename T>
  using lvalue_or_self_contained2_t = std::conditional_t<std::is_lvalue_reference_v<T>, std::decay_t<T>, self_contained_t<T>>;


  // ---------------------------------------------------------------------------------------- //


  // ---------------------------------------------- //
  //  Traits to be defined in the matrix interface  //
  // ---------------------------------------------- //

  namespace internal
  {
    // Type trait testing whether an object is a zero matrix.
#ifdef __cpp_concepts
    template<typename T>
    struct is_zero_matrix : class_trait<is_zero_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_zero_matrix : class_trait<is_zero_matrix, T, Enable> {};
#endif
  }

  /**
   * T is a zero matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = internal::is_zero_matrix<T>::value;
#else
  inline constexpr bool zero_matrix = internal::is_zero_matrix<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether an object is an identity matrix.
#ifdef __cpp_concepts
    template<typename T>
    struct is_identity_matrix : internal::class_trait<is_identity_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_identity_matrix : internal::class_trait<is_identity_matrix, T, Enable> {};
#endif
  }

  /**
   * T is an identity matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept identity_matrix = internal::is_identity_matrix<T>::value;
#else
  inline constexpr bool identity_matrix = internal::is_identity_matrix<T>::value;
#endif


#ifdef __cpp_concepts
  /**
   * T is a one-by-one matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept one_by_one_matrix = (MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1);
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_1by1 : internal::class_trait<is_1by1, T, Enable> {};

    template<typename T>
    struct is_1by1<T, std::enable_if_t<(MatrixTraits<T>::dimension == 1) and (MatrixTraits<T>::columns == 1)>>
      : std::true_type {};
  }

  /// T is a 1-by-1 matrix.
  template<typename T>
  inline constexpr bool one_by_one_matrix = detail::is_1by1<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether an object is a diagonal matrix (other than zero_matrix, identity_matrix, or one_by_one_matrix).
#ifdef __cpp_concepts
    template<typename T>
    struct is_diagonal_matrix : internal::class_trait<is_diagonal_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_diagonal_matrix : internal::class_trait<is_diagonal_matrix, T, Enable> {};
#endif
  }

#ifdef __cpp_concepts
  /**
   * T is a diagonal matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept diagonal_matrix =
    internal::is_diagonal_matrix<T>::value or zero_matrix<T> or identity_matrix<T> or one_by_one_matrix<T>;
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_diag_matrix_impl : internal::class_trait<is_diag_matrix_impl, T, Enable> {};

    template<typename T>
    struct is_diag_matrix_impl<T, std::enable_if_t<
      internal::is_diagonal_matrix<T>::value or zero_matrix<T> or identity_matrix<T> or one_by_one_matrix<T>>>
      : std::true_type {};
  }

  /// T is a diagonal matrix.
  template<typename T>
  inline constexpr bool diagonal_matrix = detail::is_diag_matrix_impl<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether an object is a self-adjoint matrix (other than diagonal_matrix).
#ifdef __cpp_concepts
    template<typename T>
    struct is_self_adjoint_matrix : internal::class_trait<is_self_adjoint_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_self_adjoint_matrix : internal::class_trait<is_self_adjoint_matrix, T, Enable> {};
#endif
  }

#ifdef __cpp_concepts
  /**
   * T is a self-adjoint matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept self_adjoint_matrix = internal::is_self_adjoint_matrix<T>::value or diagonal_matrix<T>;
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_sa_matrix_impl : internal::class_trait<is_sa_matrix_impl, T, Enable> {};

    template<typename T>
    struct is_sa_matrix_impl<T, std::enable_if_t<internal::is_self_adjoint_matrix<T>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }

  /// T is a self-adjoint matrix.
  template<typename T>
  inline constexpr bool self_adjoint_matrix = detail::is_sa_matrix_impl<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether an object is a lower-triangular matrix (other than diagonal_matrix).
#ifdef __cpp_concepts
    template<typename T>
    struct is_lower_triangular_matrix : internal::class_trait<is_lower_triangular_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_lower_triangular_matrix : internal::class_trait<is_lower_triangular_matrix, T, Enable> {};
#endif
  }

#ifdef __cpp_concepts
  /**
   * T is a lower-triangular matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept lower_triangular_matrix = internal::is_lower_triangular_matrix<T>::value or diagonal_matrix<T>;
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_lt_matrix_impl : internal::class_trait<is_lt_matrix_impl, T, Enable> {};

    template<typename T>
    struct is_lt_matrix_impl<T, std::enable_if_t<internal::is_lower_triangular_matrix<T>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }

  /// T is a lower-triangular matrix.
  template<typename T>
  inline constexpr bool lower_triangular_matrix = detail::is_lt_matrix_impl<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether an object is an upper-triangular matrix (other than diagonal_matrix).
#ifdef __cpp_concepts
    template<typename T>
    struct is_upper_triangular_matrix : internal::class_trait<is_upper_triangular_matrix, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_upper_triangular_matrix : internal::class_trait<is_upper_triangular_matrix, T, Enable> {};
#endif
  }

#ifdef __cpp_concepts
  /**
   * T is an upper-triangular matrix.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept upper_triangular_matrix = internal::is_upper_triangular_matrix<T>::value or diagonal_matrix<T>;
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_ut_matrix_impl : internal::class_trait<is_ut_matrix_impl, T, Enable> {};

    template<typename T>
    struct is_ut_matrix_impl<T, std::enable_if_t<internal::is_upper_triangular_matrix<T>::value or diagonal_matrix<T>>>
      : std::true_type {};
  }

  /// T is an upper-triangular matrix.
  template<typename T>
  inline constexpr bool upper_triangular_matrix = detail::is_ut_matrix_impl<T>::value;
#endif


#ifdef __cpp_concepts
  /**
   * T is a triangular matrix (upper or lower).
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
  concept triangular_matrix = lower_triangular_matrix<T> or upper_triangular_matrix<T>;
#else
  namespace detail
  {
    template<typename T, typename Enable = void>
    struct is_triangular_matrix_impl : internal::class_trait<is_triangular_matrix_impl, T, Enable> {};

    template<typename T>
    struct is_triangular_matrix_impl<T, std::enable_if_t<lower_triangular_matrix<T> or upper_triangular_matrix<T>>>
      : std::true_type {};
  }

  template<typename T>
  inline constexpr bool triangular_matrix = detail::is_triangular_matrix_impl<T>::value;
#endif


  namespace internal
  {
    // Type trait testing whether the base matrix is a Cholesky square root.
#ifdef __cpp_concepts
    template<typename T>
    struct is_cholesky_form : internal::class_trait<is_cholesky_form, T> {};
#else
    template<typename T, typename Enable = void>
    struct is_cholesky_form : internal::class_trait<is_cholesky_form, T, Enable> {};
#endif
  }

  /**
   * T has a base matrix that is a Cholesky square root.
   *
   * If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = internal::is_cholesky_form<T>::value;
#else
  inline constexpr bool cholesky_form = internal::is_cholesky_form<T>::value;
#endif


  /*
   * The type of a triangular matrix, either lower, upper, or diagonal.
   */
  enum struct TriangleType { lower, upper, diagonal };


  namespace detail
  {
    /// Derive TriangleType from type traits.
#ifdef __cpp_concepts
    template<triangular_matrix T>
#else
    template<typename T, typename Enable = void>
#endif
    struct triangle_type_of_impl : std::integral_constant<TriangleType, TriangleType::lower> {};


#ifdef __cpp_concepts
    template<lower_triangular_matrix T> requires (not diagonal_matrix<T>)
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<lower_triangular_matrix<T> and not diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::lower> {};


#ifdef __cpp_concepts
    template<upper_triangular_matrix T> requires (not diagonal_matrix<T>)
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<upper_triangular_matrix<T> and not diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::upper> {};


#ifdef __cpp_concepts
    template<diagonal_matrix T>
    struct triangle_type_of_impl<T>
#else
    template<typename T>
    struct triangle_type_of_impl<T, std::enable_if_t<diagonal_matrix<T>>>
#endif
      : std::integral_constant<TriangleType, TriangleType::diagonal> {};

  } // namespace detail


  /**
   * Derive the TriangleType from the type of the triangular_matrix.
   */
#ifdef __cpp_concepts
  template<triangular_matrix T>
  inline constexpr TriangleType triangle_type_of = detail::triangle_type_of_impl<T>::value;
#else
  template<typename T>
  inline constexpr TriangleType triangle_type_of = detail::triangle_type_of_impl<T>::value;
#endif


  // ------------------------------------------------- //
  //  Type traits as specialized for OpenKalman types  //
  // ------------------------------------------------- //

  namespace internal
  {
    // ---------------- //
    //  is_zero_matrix  //
    // ---------------- //

#ifdef __cpp_concepts
    template<typename T> requires typed_matrix<T> or covariance<T>
    struct is_zero_matrix<T>
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
#endif
      : is_zero_matrix<typename MatrixTraits<T>::BaseMatrix> {};


#ifdef __cpp_concepts
    template<distribution T>
    struct is_zero_matrix<T>
#else
    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<distribution<T>>>
#endif
      : std::bool_constant<zero_matrix<typename DistributionTraits<T>::Mean> and
        zero_matrix<typename DistributionTraits<T>::Covariance>> {};


    // -------------------- //
    //  is_identity_matrix  //
    // -------------------- //

#ifdef __cpp_concepts
    template<covariance T>
    struct is_identity_matrix<T>
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<covariance<T>>>
#endif
      : is_identity_matrix<typename MatrixTraits<T>::BaseMatrix> {};


#ifdef __cpp_concepts
    template<typed_matrix T> requires
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_identity_matrix<T>
#else
    template<typename T>
    struct is_identity_matrix<T, std::enable_if_t<typed_matrix<T> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
#endif
      : is_identity_matrix<typename MatrixTraits<T>::BaseMatrix> {};


    // -------------------- //
    //  is_diagonal_matrix  //
    // -------------------- //

#ifdef __cpp_concepts
    template<covariance T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<covariance<T> and diagonal_matrix<typename MatrixTraits<T>::BaseMatrix>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typed_matrix T> requires diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_diagonal_matrix<T>
#else
    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<typed_matrix<T> and
      diagonal_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<distribution T> requires diagonal_matrix<typename DistributionTraits<T>::Covariance>
    struct is_diagonal_matrix<T>
#else
      template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<distribution<T> and
      diagonal_matrix<typename DistributionTraits<T>::Covariance>>>
#endif
      : std::true_type {};


    // ------------------------ //
    //  is_self_adjoint_matrix  //
    // ------------------------ //

#ifdef __cpp_concepts
    template<covariance T> requires (not square_root_covariance<T>)
    struct is_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<covariance<T> and (not square_root_covariance<T>)>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typed_matrix T> requires self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>
    struct is_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_self_adjoint_matrix<T, std::enable_if_t<typed_matrix<T> and
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix> and
      equivalent_to<typename MatrixTraits<T>::RowCoefficients, typename MatrixTraits<T>::ColumnCoefficients>>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<distribution T> requires self_adjoint_matrix<typename DistributionTraits<T>::Covariance>
    struct is_self_adjoint_matrix<T>
#else
    template<typename T>
    struct is_self_adjoint_matrix<T,
      std::enable_if_t<distribution<T> and self_adjoint_matrix<typename DistributionTraits<T>::Covariance>>>
#endif
      : std::true_type {};


    // ------------------ //
    //  is_cholesky_form  //
    // ------------------ //

#ifdef __cpp_concepts
    template<covariance T> requires (not self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>)
    struct is_cholesky_form<T>
#else
    template<typename T>
    struct is_cholesky_form<T,
      std::enable_if_t<covariance<T> and (not self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>)>>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<distribution T> requires cholesky_form<typename DistributionTraits<T>::Covariance>
    struct is_cholesky_form<T>
#else
    template<typename T>
    struct is_cholesky_form<T,
      std::enable_if_t<distribution<T> and cholesky_form<typename DistributionTraits<T>::Covariance>>>
#endif
      : std::true_type {};


    // ---------------------------- //
    //  is_lower_triangular_matrix  //
    // ---------------------------- //

#ifdef __cpp_concepts
    template<square_root_covariance T> requires lower_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_lower_triangular_matrix<T>
#else
    template<typename T>
    struct is_lower_triangular_matrix<T,
      std::enable_if_t<square_root_covariance<T> and lower_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>>>
#endif
      : std::true_type {};


    // ---------------------------- //
    //  is_upper_triangular_matrix  //
    // ---------------------------- //

#ifdef __cpp_concepts
    template<square_root_covariance T> requires upper_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>
    struct is_upper_triangular_matrix<T>
#else
    template<typename T>
    struct is_upper_triangular_matrix<T, std::enable_if_t<square_root_covariance<T> and
      upper_triangular_matrix<typename MatrixTraits<T>::BaseMatrix> or
      self_adjoint_matrix<typename MatrixTraits<T>::BaseMatrix>>>
#endif
      : std::true_type {};


    // ------------------- //
    //  is_self_contained  //
    // ------------------- //

#ifdef __cpp_concepts
    template<typename T> requires typed_matrix<T> or covariance<T>
    struct is_self_contained<T>
#else
    template<typename T>
    struct is_self_contained<T, std::enable_if_t<typed_matrix<T> or covariance<T>>>
#endif
      : is_self_contained<typename MatrixTraits<T>::BaseMatrix> {};


#ifdef __cpp_concepts
    template<distribution T>
    struct is_self_contained<T>
#else
      template<typename T>
    struct is_self_contained<T, std::enable_if_t<distribution<T>>>
#endif
      : std::bool_constant<self_contained<typename DistributionTraits<T>::Mean> and
        self_contained<typename DistributionTraits<T>::Covariance>> {};




  } // namespace internal


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
