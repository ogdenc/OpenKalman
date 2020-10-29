/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FORWARD_DECLARATIONS_HPP
#define OPENKALMAN_FORWARD_DECLARATIONS_HPP

namespace OpenKalman
{
  // ------------ //
  //   Matrices   //
  // ------------ //

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
  template<coefficients RowCoefficients, coefficients ColumnCoefficients, typename ArgType> requires
    is_typed_matrix_base_v<ArgType> and (RowCoefficients::size == MatrixTraits<ArgType>::dimension) and
    (ColumnCoefficients::size == MatrixTraits<ArgType>::columns)
#else
  template<typename RowCoefficients, typename ColumnCoefficients, typename ArgType>
#endif
  struct Matrix;

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
  struct is_typed_matrix<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>> : std::true_type {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
  struct is_zero<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>> : is_zero<BaseMatrix> {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
  struct is_identity<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>>
    : std::bool_constant<is_identity_v<BaseMatrix> and
    is_equivalent_v<RowCoefficients, ColumnCoefficients>> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_diagonal<Matrix<Coefficients, Coefficients, BaseMatrix>,
    std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix> and not is_1by1_v<BaseMatrix>>>
    : is_diagonal<BaseMatrix> {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix>
  struct is_strict<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};


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
  template<coefficients Coefficients, typename BaseMatrix> requires
    is_typed_matrix_base_v<BaseMatrix> and (Coefficients::size == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct Mean;

  template<typename Coefficients, typename BaseMatrix>
  struct is_mean<Mean<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix<Mean<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_zero<Mean<Coefficients, BaseMatrix>> : is_zero<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_identity<Mean<Coefficients, BaseMatrix>>
    : std::bool_constant<is_identity_v<BaseMatrix> and Coefficients::axes_only> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Mean<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Mean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Mean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};

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
  template<coefficients Coefficients, typename BaseMatrix> requires
  is_typed_matrix_base_v<BaseMatrix> and (Coefficients::dimension == MatrixTraits<BaseMatrix>::dimension)
#else
  template<typename Coefficients, typename BaseMatrix>
#endif
  struct EuclideanMean;

  template<typename Coefficients, typename BaseMatrix>
  struct is_Euclidean_mean<EuclideanMean<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_typed_matrix<EuclideanMean<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_zero<EuclideanMean<Coefficients, BaseMatrix>> : is_zero<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_identity<EuclideanMean<Coefficients, BaseMatrix>>
    : std::bool_constant<OpenKalman::is_identity_v<BaseMatrix> and Coefficients::axes_only> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<EuclideanMean<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<EuclideanMean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<EuclideanMean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};


  /**
   * @brief A Covariance matrix.
   * @tparam Coefficients Coefficient types.
   * @tparam ArgType Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
  template<
    typename Coefficients,
    typename ArgType>
  struct Covariance;

  template<typename Coefficients, typename BaseMatrix>
  struct is_covariance<Covariance<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_Cholesky<Covariance<Coefficients, BaseMatrix>>
    : std::bool_constant<is_triangular_v<BaseMatrix> and not is_self_adjoint_v<BaseMatrix>> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_self_adjoint<Covariance<Coefficients, BaseMatrix>, std::enable_if_t<not OpenKalman::is_diagonal_v<BaseMatrix>>>
    : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_diagonal<Covariance<Coefficients, BaseMatrix>,
    std::enable_if_t<not OpenKalman::is_zero_v<BaseMatrix> and not OpenKalman::is_identity_v<BaseMatrix>>>
    : is_diagonal<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_identity<Covariance<Coefficients, BaseMatrix>, std::enable_if_t<not is_zero_v<BaseMatrix>>>
    : is_identity<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_zero<Covariance<Coefficients, BaseMatrix>> : is_zero<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<Covariance<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Covariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Covariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};

  /**
   * @brief The upper or lower triangle Cholesky factor (square root) of a covariance matrix.
   * If S is a SquareRootCovariance, S*S.transpose() is a Covariance
   * @tparam Coefficients Coefficient types.
   * @tparam ArgType Type of the underlying storage matrix (e.g., self-adjoint or triangular).
   */
  template<
    typename Coefficients,
    typename ArgType>
 struct SquareRootCovariance;

  template<typename Coefficients, typename BaseMatrix>
  struct is_covariance<SquareRootCovariance<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_square_root<SquareRootCovariance<Coefficients, BaseMatrix>> : std::true_type {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_Cholesky<SquareRootCovariance<Coefficients, BaseMatrix>>
    : std::bool_constant<OpenKalman::is_triangular_v<BaseMatrix> and not OpenKalman::is_self_adjoint_v<BaseMatrix>> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_lower_triangular<SquareRootCovariance<Coefficients, BaseMatrix>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::integral_constant<bool,
    is_lower_triangular_v<typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_upper_triangular<SquareRootCovariance<Coefficients, BaseMatrix>,
    std::enable_if_t<not is_diagonal_v<BaseMatrix>>>
    : std::integral_constant<bool,
    is_upper_triangular_v<typename MatrixTraits<BaseMatrix>::template TriangularBaseType<>>> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_diagonal<SquareRootCovariance<Coefficients, BaseMatrix>,
  std::enable_if_t<not is_zero_v<BaseMatrix> and not is_identity_v<BaseMatrix>>>
  : is_diagonal<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_identity<SquareRootCovariance<Coefficients, BaseMatrix>, std::enable_if_t<not is_zero_v<BaseMatrix>>>
  : is_identity<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_zero<SquareRootCovariance<Coefficients, BaseMatrix>> : is_zero<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix>
  struct is_strict<SquareRootCovariance<Coefficients, BaseMatrix>> : is_strict<BaseMatrix> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<SquareRootCovariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<SquareRootCovariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};


  namespace internal
  {
    /// Convert a covariance to a base matrix of a particular type
    template<typename T = void, typename Arg>
    constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    template<typename Derived, typename ArgType>
    struct MatrixBase;

    /// A helper object for setting elements of a matrix.
    template<bool read_only, typename T>
    struct ElementSetter;

    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});

    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});
  }


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

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_Gaussian_distribution<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
    : std::true_type {};

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_Cholesky<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
    : std::bool_constant<not is_self_adjoint_v<CovarianceMatrix>> {};

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_self_adjoint<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>,
    std::enable_if_t<not is_diagonal_v<CovarianceMatrix>>>
    : std::true_type {};

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_diagonal<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>,
    std::enable_if_t<not is_zero_v<MeanMatrix> or not is_zero_v<CovarianceMatrix>>>
    : is_diagonal<CovarianceMatrix> {};

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_zero<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
    : std::bool_constant<is_zero_v<MeanMatrix> and is_zero_v<CovarianceMatrix>> {};

  template<typename Coefficients, typename MeanMatrix, typename CovarianceMatrix, typename re>
  struct is_strict<GaussianDistribution<Coefficients, MeanMatrix, CovarianceMatrix, re>>
    : std::bool_constant<is_strict_v<MeanMatrix> and is_strict_v<CovarianceMatrix>> {};


}

#endif //OPENKALMAN_FORWARD_DECLARATIONS_HPP
