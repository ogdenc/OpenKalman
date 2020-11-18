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
 * \file Traits.h
 * A header file containing forward declarations for all OpenKalman traits.
 */

#ifndef OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP
#define OPENKALMAN_FORWARD_CLASS_DECLARATIONS_HPP

#include <type_traits>

namespace OpenKalman
{
  /**
   * A matrix with typed rows and columns.
   *
   * The matrix can be thought of as a transformation from X to Y, where the coefficients for each of X and Y are typed.
   * Example declaration:
   * <code>Matrix<double, Coefficients<Axis, Axis, Angle>, Coefficients<Axis, Axis>> x;</code>
   * \tparam RowCoefficients A set of coefficients (e.g., Axis, Spherical, etc.) corresponding to the rows.
   * \tparam ColumnCoefficients Another set of coefficients corresponding to the columns.
   * \tparam ArgType The base matrix type.
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
   * \brief A set of column vectors representing one or more means.
   * Generally, a column vector representing a mean. Alternatively, it can be a 2D matrix representing a collection of
   * column vectors of the same coefficient types, each column vector representing a distinct mean.
   * Example declaration:
   * <code>Mean<Coefficients<Axis, Axis, Angle>, 1, Eigen::Matrix<double, 3, 1>> x;</code>
   * This declares a 3-dimensional vector <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an Angle, all of scalar type <code>double</code>. The underlying representation is an
   * Eigen3 column vector.
   * \tparam Coefficients Coefficient types of the mean (e.g., Axis, Polar).
   * \tparam BaseMatrix Regular matrix on which the mean is based (usually a column vector).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct Mean;


  /**
   * \brief The underlying class representing the Euclidean space version of a mean, with typed coefficients.
   *
   * Example declaration:
   * <code>EuclideanMean<Coefficients<Axis, Axis, Angle>, 1, Eigen::Matrix<double, 3, 1>> x;</code>
   * This declares a 3-dimensional mean <var>x</var>, where the coefficients are, respectively, an Axis,
   * an Axis, and an Angle, all of scalar type <code>double</code>. The underlying representation is a
   * four-dimensional vector in Euclidean space, with two of the dimensions representing the Angle coefficient.
   * \tparam Coefficients A set of coefficients (e.g., Angle, Polar, etc.)
   * \tparam BaseMatrix The mean's base type. This is a column vector or a matrix (considered as a collection of column vectors).
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
   * \tparam Coefficients Coefficient types.
   * \tparam BaseMatrix Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct Covariance;


  /**
   * \brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * \details If S is a SquareRootCovariance, S*S.transpose() is a Covariance.
   * If BaseMatrix is triangular, the SquareRootCovariance has the same triangle type (upper or lower). If BaseMatrix
   * is self-adjoint, the triangle type of SquareRootCovariance is considered either upper ''or'' lower.
   * \tparam Coefficients Coefficient types.
   * \tparam BaseMatrix Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, covariance_base BaseMatrix> requires
  (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct SquareRootCovariance;


  /**
   * \brief A Gaussian distribution, defined in terms of a mean vector and a covariance matrix.
   * \tparam Coefficients Coefficient types.
   * \tparam ArgMean Underlying type for Mean.
   * \tparam ArgMoment Underlying type for Moment.
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
    struct is_mean : std::false_type {};

    template<typename Coefficients, typename BaseMatrix>
    struct is_mean<Mean<Coefficients, BaseMatrix>> : std::true_type {};
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

    template<typename Coefficients, typename BaseMatrix>
    struct is_euclidean_mean<EuclideanMean<Coefficients, BaseMatrix>> : std::true_type {};
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
   * T is a euclidean_mean that actually has coefficients that are transformed to Euclidean space.
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

    template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
    struct is_matrix<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>> : std::true_type {};
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

    template<typename Coefficients, typename BaseMatrix>
    struct is_square_root_covariance<SquareRootCovariance<Coefficients, BaseMatrix>> : std::true_type {};
  }


  /**
   * T is a square root (Cholesky) covariance matrix with typed rows and columns. The rows and columns have the same type.
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

    template<typename Coefficients, typename BaseMatrix>
    struct is_sa_covariance<Covariance<Coefficients, BaseMatrix>> : std::true_type {};
  }


  /**
   * T is a covariance matrix of any kind, including a square_root_covariance. The rows and columns have the same type.
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

    template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
    struct is_gaussian_distribution<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
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
      requires (std::is_void_v<T> or covariance_base<T>) and (covariance<Arg> or typed_matrix<Arg>)
#endif
    constexpr decltype(auto)
    convert_base_matrix(Arg&&) noexcept;


    // Definition and documentation are in MatrixBase.hpp
    template<typename Derived, typename ArgType>
    struct MatrixBase;


    // Definition and documentation are in CovarianceBase.hpp.
#ifdef __cpp_concepts
    template<typename Derived, typename ArgType>
#else
    template<typename Derived, typename ArgType, typename Enable = void>
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
