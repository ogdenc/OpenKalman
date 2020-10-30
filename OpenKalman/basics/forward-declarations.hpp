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


  // --------- //
  //   Means   //
  // --------- //

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
    // Convert a covariance to a base matrix of a particular type
    template<typename T = void, typename Arg>
    constexpr decltype(auto) convert_base_matrix(Arg&&) noexcept;

    template<typename Derived, typename ArgType>
    struct MatrixBase;

    // Base of Covariance and SquareRootCovariance classes.
    template<typename Derived, typename ArgType, typename Enable = void>
    struct CovarianceBase;

    // A helper object for setting elements of a matrix.
    template<bool read_only, typename T>
    struct ElementSetter;

    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});

    template<bool read_only, typename T>
    auto make_ElementSetter(T&&, std::size_t,
      const std::function<void()>& = []{}, const std::function<void()>& = []{});
  }


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
