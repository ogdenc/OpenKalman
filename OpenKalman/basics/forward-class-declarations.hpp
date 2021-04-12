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
   * The matrix can be thought of as a transformation from X to Y, where the coefficients for each of X and Y are typed.
   * Example declarations:
   * - <code>Matrix<Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>,
   * native_matrix_t<native_matrix_t<double, 3, 2>> x;</code>
   * - <code>Matrix<double, Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>,
   * native_matrix_t<double, 3, 2>> x;</code>
   * \tparam RowCoefficients A set of \ref OpenKalman::coefficients "coefficients" (e.g., Axis, Spherical, etc.)
   * corresponding to the rows.
   * \tparam ColumnCoefficients Another set of \ref OpenKalman::coefficients "coefficients" corresponding
   * to the columns.
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable NestedMatrix> requires
    (RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and
    (ColumnCoefficients::dimensions == MatrixTraits<NestedMatrix>::columns) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
#endif
  struct Matrix;


  /**
   * \brief A set of one or more column vectors, each representing a statistical mean.
   * \details Unlike OpenKalman::Matrix, the columns of a Mean are untyped. When a Mean is converted to an
   * OpenKalman::Matrix, the columns are assigned type Axis.
   * Example declaration:
   * <code>Mean<Coefficients<Axis, Axis, angle::Radians>, 1, native_matrix_t<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * \tparam Coefficients Coefficient types of the mean (e.g., Axis, Polar).
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, typed_matrix_nestable NestedMatrix> requires
  (RowCoefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and
  (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename RowCoefficients, typename NestedMatrix>
#endif
  struct Mean;


  /**
   * \brief Similar to a Mean, but the coefficients are transformed into Euclidean space, based on their type.
   * \details Means containing angles should be converted to EuclideanMean before taking an average or weighted average.
   * Example declaration:
   * <code>EuclideanMean<Coefficients<Axis, Axis, angle::Radians>, 1, native_matrix_t<double, 4, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an angle::Radians, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with the last two of the dimensions representing the angle::Radians coefficient
   * transformed to x and y locations on a unit circle associated with the angle::Radians-type coefficient.
   * \tparam Coefficients A set of coefficients (e.g., Axis, angle::Radians, Polar, etc.)
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable NestedMatrix> requires
  (Coefficients::euclidean_dimensions == MatrixTraits<NestedMatrix>::rows) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct EuclideanMean;


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
    (Coefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct Covariance;


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
    (Coefficients::dimensions == MatrixTraits<NestedMatrix>::rows) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
#endif
  struct SquareRootCovariance;


  /**
   * \brief A Gaussian distribution, defined in terms of a Mean and a Covariance.
   * \tparam Coefficients Coefficient types.
   * \tparam MeanNestedMatrix The underlying native matrix for the Mean.
   * \tparam CovarianceNestedMatrix The underlying native matrix (triangular or self-adjoint) for the Covariance.
   * \tparam random_number_engine A random number engine compatible with the c++ standard library (e.g., std::mt19937).
   */
#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine = std::mt19937> requires
      (MatrixTraits<MeanNestedMatrix>::rows == MatrixTraits<CovarianceNestedMatrix>::rows) and
      (MatrixTraits<MeanNestedMatrix>::columns == 1) and
      (std::is_same_v<typename MatrixTraits<MeanNestedMatrix>::Scalar,
        typename MatrixTraits<CovarianceNestedMatrix>::Scalar>)
#else
  template<
    typename Coefficients,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine = std::mt19937>
#endif
  struct GaussianDistribution;


  // --------- //
  //   Means   //
  // --------- //


  namespace detail
  {
    template<typename T>
    struct is_mean : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_mean<Mean<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a mean (i.e., is a specialization of the class Mean).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept mean = detail::is_mean<std::decay_t<T>>::value;
#else
  inline constexpr bool mean = detail::is_mean<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_wrapped_mean : std::false_type {};

    template<typename T>
    struct is_wrapped_mean<T, std::enable_if_t<mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a wrapped mean (i.e., its row coefficients have at least one type that requires wrapping).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept wrapped_mean = mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  template<typename T>
  inline constexpr bool wrapped_mean = detail::is_wrapped_mean<T>::value;
#endif


  // ------------------- //
  //   Euclidean means   //
  // ------------------- //

  namespace detail
  {
    template<typename T>
    struct is_euclidean_mean : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_euclidean_mean<EuclideanMean<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a Euclidean mean (i.e., is a specialization of the class EuclideanMean).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_mean = detail::is_euclidean_mean<std::decay_t<T>>::value;
#else
  inline constexpr bool euclidean_mean = detail::is_euclidean_mean<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_euclidean_transformed : std::false_type {};

    template<typename T>
    struct is_euclidean_transformed<T, std::enable_if_t<
      euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a Euclidean mean that actually has coefficients that are transformed to Euclidean space.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_transformed = euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only);
#else
  template<typename T>
  inline constexpr bool euclidean_transformed = detail::is_euclidean_transformed<T>::value;
#endif


  // ------------------ //
  //   typed matrices   //
  // ------------------ //

  namespace detail
  {
    template<typename T>
    struct is_matrix : std::false_type {};

    template<typename RowCoefficients, typename ColumnCoefficients, typename NestedMatrix>
    struct is_matrix<Matrix<RowCoefficients, ColumnCoefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief Specifies that T is a typed matrix (i.e., is a specialization of Matrix, Mean, or EuclideanMean).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix = mean<T> or euclidean_mean<T> or detail::is_matrix<std::decay_t<T>>::value;
#else
  inline constexpr bool typed_matrix = mean<T> or euclidean_mean<T> or detail::is_matrix<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_untyped_columns : std::false_type {};

    template<typename T>
    struct has_untyped_columns<T, std::enable_if_t<typed_matrix<T> and MatrixTraits<T>::ColumnCoefficients::axes_only>>
      : std::true_type {};

    template<typename T>
    struct has_untyped_columns<T, std::enable_if_t<
      (not typed_matrix<T>) and std::is_integral_v<decltype(MatrixTraits<T>::columns)>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T has untyped (or Axis typed) column coefficients.
   * \details T must be either a native matrix or its columns must all have type Axis.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept untyped_columns = (typed_matrix<T> and MatrixTraits<T>::ColumnCoefficients::axes_only) or
    (not typed_matrix<T> and requires {MatrixTraits<T>::columns;});
#else
  template<typename T>
  inline constexpr bool untyped_columns = detail::has_untyped_columns<std::decay_t<T>>::value;
#endif


#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_one_column : std::false_type {};

    template<typename T>
    struct has_one_column<T, std::enable_if_t<MatrixTraits<T>::columns == 1>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T is a column vector (i.e., has one untyped or Axis-typed column).
   * \details If T is a typed_matrix, its column must be of type Axis.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept column_vector = untyped_columns<T> and (MatrixTraits<T>::columns == 1);
#else
  template<typename T>
  inline constexpr bool column_vector = untyped_columns<T> and detail::has_one_column<std::decay_t<T>>::value;
#endif


  // ------------------------------------ //
  //  square root (Cholesky) covariances  //
  // ------------------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_square_root_covariance : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_square_root_covariance<SquareRootCovariance<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief T is a square root (Cholesky) covariance matrix (i.e., a specialization of SquareRootCovariance).
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept square_root_covariance = detail::is_square_root_covariance<std::decay_t<T>>::value;
#else
  inline constexpr bool square_root_covariance = detail::is_square_root_covariance<std::decay_t<T>>::value;
#endif


  // ------------------------ //
  //  covariances in general  //
  // ------------------------ //

  namespace detail
  {
    template<typename T>
    struct is_sa_covariance : std::false_type {};

    template<typename Coefficients, typename NestedMatrix>
    struct is_sa_covariance<Covariance<Coefficients, NestedMatrix>> : std::true_type {};
  }


  /**
   * \brief T is a specialization of either Covariance or SquareRootCovariance.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance = square_root_covariance<T> or detail::is_sa_covariance<std::decay_t<T>>::value;
#else
  inline constexpr bool covariance = square_root_covariance<T> or detail::is_sa_covariance<std::decay_t<T>>::value;
#endif


  // --------------- //
  //  distributions  //
  // --------------- //

  namespace detail
  {
    template<typename T>
    struct is_gaussian_distribution : std::false_type {};

    template<typename Coefficients, typename MeanNestedMatrix, typename CovarianceNestedMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<Coefficients, MeanNestedMatrix, CovarianceNestedMatrix, re>>
      : std::true_type {};
  }

  /**
   * \brief T is a Gaussian distribution.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = detail::is_gaussian_distribution<std::decay_t<T>>::value;
#else
  inline constexpr bool gaussian_distribution = detail::is_gaussian_distribution<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a statistical distribution of any kind that is defined in OpenKalman.
   * \note This is a concept when compiled with c++20, and a constexpr bool in c++17.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  inline constexpr bool distribution = gaussian_distribution<T>;
#endif


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
     * \brief An interface to a matrix, to be used for getting and setting the individual matrix elements.
     * \tparam read_only Whether the matrix elements are read-only (as opposed to writable).
     * \tparam T The matrix type.
     */
    template<bool read_only, typename T>
    struct ElementSetter;


    /**
     * \internal
     * \brief Make an ElementSetter that takes two indices.
     * \tparam read_only Whether the matrix elements are read-only.
     * \tparam T The matrix type
     * \param do_before The action, if any, taken before getting or setting an element.
     * \param on_set The action, if any, taken after setting an element.
     */
    template<bool read_only, typename T>
    auto make_ElementSetter(T&& t, std::size_t i, std::size_t j,
      const std::function<void()>& do_before = []{}, const std::function<void()>& on_set = []{});


    /**
     * \internal
     * \brief Make an ElementSetter that takes one index (i.e., is a vector).
     * \tparam read_only Whether the matrix elements are read-only.
     * \tparam T The matrix type
     * \param do_before The action, if any, taken before getting or setting an element.
     * \param on_set The action, if any, taken after setting an element.
     */
    template<bool read_only, typename T>
    auto make_ElementSetter(T&& t, std::size_t i,
      const std::function<void()>& do_before = []{}, const std::function<void()>& on_set = []{});

  } // namespace internal


} // OpenKalman

#endif //OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
