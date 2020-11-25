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
 * \file forward-class-declarations.hpp
 * A header file containing forward declarations for OpenKalman classes and some of their closely-associated traits.
 */

#ifndef OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
#define OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * \brief A matrix with typed rows and columns.
   * \details It is a wrapper for a native matrix type from a supported matrix library such as Eigen.
   * The matrix can be thought of as a transformation from X to Y, where the coefficients for each of X and Y are typed.
   * Example declarations:
   * - <code>Matrix<Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>, native_matrix_t<native_matrix_t<double, 3, 2>> x;</code>
   * - <code>Matrix<double, Coefficients<Axis, Axis, angle::Radians>, Coefficients<Axis, Axis>,
   * native_matrix_t<double, 3, 2>> x;</code>
   * \tparam RowCoefficients A set of coefficients (e.g., Axis, Spherical, etc.) corresponding to the rows.
   * \tparam ColumnCoefficients Another set of coefficients corresponding to the columns.
   * \tparam NestedMatrix The underlying native matrix or matrix expression.
   */
#ifdef __cpp_concepts
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typed_matrix_nestable NestedMatrix> requires
    (RowCoefficients::size == MatrixTraits<NestedMatrix>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<NestedMatrix>::columns) and (not std::is_rvalue_reference_v<NestedMatrix>)
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
  template<coefficients Coefficients, typed_matrix_nestable NestedMatrix> requires
  (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
#else
  template<typename Coefficients, typename NestedMatrix>
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
  (Coefficients::dimension == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
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
    (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
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
  (Coefficients::size == MatrixTraits<NestedMatrix>::dimension) and (not std::is_rvalue_reference_v<NestedMatrix>)
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
  template<
    typename Coefficients,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine> // = std::mt19937
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
   * T is a mean (i.e., is a specialization of the class Mean).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
    template<typename T, typename Enable = void>
    struct is_wrapped_mean : std::false_type {};

    template<typename T>
    struct is_wrapped_mean<T, std::enable_if_t<mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * T is a wrapped mean (i.e., its row coefficients have at least one type that requires wrapping).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
   * T is a Euclidean mean (i.e., is a specialization of the class EuclideanMean).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
    template<typename T, typename Enable = void>
    struct is_euclidean_transformed : std::false_type {};

    template<typename T>
    struct is_euclidean_transformed<T, std::enable_if_t<
      euclidean_mean<T> and (not MatrixTraits<T>::RowCoefficients::axes_only)>>
      : std::true_type {};
  }
#endif


  /**
   * T is a Euclidean mean that actually has coefficients that are transformed to Euclidean space.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
   * T is a typed matrix (i.e., is a specialization of Matrix, Mean, or EuclideanMean).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
    template<typename T, typename Enable = void>
    struct is_column_vector : std::false_type {};

    template<typename T>
    struct is_column_vector<T, std::enable_if_t<typed_matrix<T> and MatrixTraits<T>::ColumnCoefficients::axes_only>>
      : std::true_type {};
  }
#endif


  /**
   * T is a column vector or set of column vectors (i.e., the columns all have type Axis).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept column_vector = typed_matrix<T> and MatrixTraits<T>::ColumnCoefficients::axes_only;
#else
  template<typename T>
  inline constexpr bool column_vector = detail::is_column_vector<std::decay_t<T>>::value;
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
   * T is a square root (Cholesky) covariance matrix (i.e., a specialization of SquareRootCovariance).
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
   * T is a specialization of either Covariance or SquareRootCovariance.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
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
   * T is a Gaussian distribution.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = detail::is_gaussian_distribution<std::decay_t<T>>::value;
#else
  inline constexpr bool gaussian_distribution = detail::is_gaussian_distribution<std::decay_t<T>>::value;
#endif


  /**
   * T is a statistical distribution of any kind that is defined in OpenKalman.
   * \note If compiled in c++17 mode, this is an inline constexpr bool variable rather than a concept.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  inline constexpr bool distribution = gaussian_distribution<T>;
#endif


  namespace internal
  {
    // Definition and documentation are in ConvertBaseMatrix.hpp
    template<typename T = void, typename Arg>
#ifdef __cpp_concepts
      requires (std::is_void_v<T> or covariance_nestable<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    constexpr decltype(auto)
    convert_nested_matrix(Arg&&) noexcept;


    // Definition and documentation are in MatrixBase.hpp
    template<typename Derived, typename NestedMatrix>
    struct MatrixBase;


    // Definition and documentation are in CovarianceBase.hpp.
#ifdef __cpp_concepts
    template<typename Derived, typename NestedMatrix>
#else
    template<typename Derived, typename NestedMatrix, typename Enable = void>
#endif
    struct CovarianceBase;


    // Definition and documentation are in ElementSetter.hpp.
    template<bool read_only, typename T>
    struct ElementSetter;


    // Definition and documentation are in ElementSetter.hpp.
    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});


    // Definition and documentation are in ElementSetter.hpp.
    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});

  } // namespace internal


} // OpenKalman

#endif //OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
