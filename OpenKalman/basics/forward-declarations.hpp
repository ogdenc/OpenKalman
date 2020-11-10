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

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename RowCoefficients, typename ColumnCoefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Matrix<RowCoefficients, ColumnCoefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};


  // --------- //
  //   Means   //
  // --------- //

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Mean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Mean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<EuclideanMean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<EuclideanMean<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};



  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_gettable<Covariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_gettable_v<BaseMatrix, N>> {};

  template<typename Coefficients, typename BaseMatrix, std::size_t N>
  struct is_element_settable<Covariance<Coefficients, BaseMatrix>, N>
    : std::bool_constant<is_element_settable_v<BaseMatrix, N>> {};


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



}

#endif //OPENKALMAN_FORWARD_DECLARATIONS_HPP
