/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_DISTRIBUTIONTRAITS_H
#define OPENKALMAN_DISTRIBUTIONTRAITS_H

#include <iostream>
#include "variables/support/TypedMatrixOverloads.h"
#include "variables/support/CovarianceOverloads.h"
#include "variables/support/OpenKalman-coefficients.h"

namespace OpenKalman
{
  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename MatrixBase, typename CovarianceBase, typename T>
  struct DistributionTraits<OpenKalman::GaussianDistribution<Coeffs, MatrixBase, CovarianceBase>, T>
  {
    using type = T;
    using Coefficients = Coeffs;
    static constexpr auto dimension = Coefficients::size;
    using Mean = Mean<Coefficients, MatrixBase>;
    using Covariance = Covariance<Coefficients, CovarianceBase>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;

    template<typename Mean, typename Covariance,
      std::enable_if_t<is_mean_v<Mean> and MatrixTraits<Mean>::columns == 1 and is_covariance_v<Covariance>, int> = 0>
    static auto make(Mean&& mean, Covariance&& covariance) noexcept
    {
      return GaussianDistribution(std::forward<Mean>(mean), std::forward<Covariance>(covariance));
    }

    static auto zero() { return make(MatrixTraits<Mean>::zero(), MatrixTraits<Covariance>::zero()); }

    static auto identity() { return make(MatrixTraits<Mean>::zero(), MatrixTraits<Covariance>::identity()); }
  };


  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
  constexpr decltype(auto)
  mean(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).mean();
  }


  template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
  constexpr decltype(auto)
  covariance(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).covariance();
  }


  template<TriangleType triangle_type = TriangleType::lower, typename Arg,
    std::enable_if_t<is_Gaussian_distribution_v<Arg> and not is_Cholesky_v<Arg>, int> = 0>
  inline auto
  to_Cholesky(Arg&& arg) noexcept
  {
    auto cov = to_Cholesky<triangle_type>(covariance(arg));
    return DistributionTraits<Arg>::make(mean(std::forward<Arg>(arg)), std::move(cov));
  }


  template<typename Arg,
    std::enable_if_t<is_Gaussian_distribution_v<Arg> and is_Cholesky_v<Arg>, int> = 0>
  inline auto
  from_Cholesky(Arg&& arg) noexcept
  {
    auto cov = from_Cholesky(covariance(arg));
    return DistributionTraits<Arg>::make(mean(std::forward<Arg>(arg)), std::move(cov));
  }


  /// Convert to strict version of the distribution.
  template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
  constexpr decltype(auto)
  strict(Arg&& arg) noexcept
  {
    using Mean = typename DistributionTraits<Arg>::Mean;
    using Covariance = typename DistributionTraits<Arg>::Covariance;
    if constexpr(is_strict_v<Mean> and is_strict_v<Covariance>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return DistributionTraits<Arg>::make(strict(mean(arg)), strict(covariance(arg)));
    }
  }


  template<typename D, typename ... Ds,
    std::enable_if_t<std::conjunction_v<is_Gaussian_distribution<D>, is_Gaussian_distribution<Ds>...>, int> = 0>
  constexpr decltype(auto)
  concatenate(const D& d, const Ds& ... ds)
  {
    if constexpr(sizeof...(Ds) > 0)
    {
      auto mean = concatenate(OpenKalman::mean(d), OpenKalman::mean(ds)...);
      auto covariance = concatenate(OpenKalman::covariance(d), OpenKalman::covariance(ds)...);
      return DistributionTraits<D>::template make(std::move(mean), std::move(covariance));
    }
    else
    {
      return std::forward<D>(d);
    }
  }


  /// Split distribution.
  template<typename C, typename ... Cs, typename D, std::enable_if_t<is_Gaussian_distribution_v<D>, int> = 0>
  inline auto
  split(const D& d) noexcept
  {
    static_assert(OpenKalman::is_equivalent_v<Concatenate<C, Cs...>, typename DistributionTraits<D>::Coefficients>);
    if constexpr(sizeof...(Cs) > 0)
    {
      using C_all = Concatenate<C, Cs...>;
      using C1 = typename C_all::template Take<C::size>;
      using C2 = typename C_all::template Discard<C::size>;
      auto[mUL, mLR] = split<C1, C2>(mean(d));
      auto[cUL, cLR] = split<C1, C2>(covariance(d));
      auto dUL = DistributionTraits<D>::template make(std::move(mUL), std::move(cUL));
      auto dLR = DistributionTraits<D>::template make(std::move(mLR), std::move(cLR));
      return std::tuple_cat(std::tuple(std::move(dUL)), split<Cs...>(std::move(dLR)));
    }
    else
    {
      return std::tuple(d);
    }
  }


  template<typename Dist, std::enable_if_t<is_Gaussian_distribution_v<Dist>, int> = 0>
  inline std::ostream&
  operator<<(std::ostream& os, const Dist& d)
  {
    os << "mean:" << std::endl << mean(d) << std::endl <<
    "covariance:" << std::endl << covariance(d) << std::endl;
    return os;
  }


  /**********************
  * Arithmetic Operators
  **********************/

  template<
    typename Dist1,
    typename Dist2,
    std::enable_if_t<is_Gaussian_distribution_v<Dist1> and is_Gaussian_distribution_v<Dist2> and
      is_equivalent_v<typename DistributionTraits<Dist1>::Coefficients,
        typename DistributionTraits<Dist2>::Coefficients>, int> = 0>
  inline auto
  operator+(const Dist1& d1, const Dist2& d2)
  {
    auto m1 = mean(d1) + mean(d2);
    auto m2 = covariance(d1) + covariance(d2);
    return DistributionTraits<Dist1>::make(std::move(m1), std::move(m2));
  };


  /// Add, to a mean vector, stochastic noise from a distribution.
  template<typename M, typename D,
    std::enable_if_t<is_column_vector_v<M> and MatrixTraits<M>::columns == 1 and
      is_Gaussian_distribution_v<D>, int> = 0>
  inline auto operator+(M&& m, D&& d)
  {
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, typename DistributionTraits<D>::Coefficients>);
    return strict(std::forward<M>(m) + std::forward<D>(d)());
  }


  template<
    typename Dist1,
    typename Dist2,
    std::enable_if_t<is_Gaussian_distribution_v<Dist1> and is_Gaussian_distribution_v<Dist2> and
      is_equivalent_v<typename DistributionTraits<Dist1>::Coefficients,
        typename DistributionTraits<Dist2>::Coefficients>, int> = 0>
  inline auto
  operator-(const Dist1& d1, const Dist2& d2)
  {
    auto m1 = mean(d1) - mean(d2);
    auto m2 = covariance(d1) - covariance(d2);
    return DistributionTraits<Dist1>::make(std::move(m1), std::move(m2));
  };


  template<
    typename A, typename D,
    std::enable_if_t<is_typed_matrix_v<A> and is_Gaussian_distribution_v<D>, int> = 0>
  inline auto
  operator*(A&& a, D&& d)
  {
    static_assert(not is_Euclidean_transformed_v<A>);
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<A>::ColumnCoefficients, typename DistributionTraits<D>::Coefficients>);
    auto m = a * mean(d);
    auto c = scale(covariance(d), a);
    return DistributionTraits<D>::make(std::move(m), std::move(c));
  }


  template<
    typename Dist, typename S,
    std::enable_if_t<is_Gaussian_distribution_v<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator*(Dist&& d, const S s)
  {
    auto m = mean(d) * s;
    auto c = scale(covariance(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };


  template<
    typename Dist, typename S,
    std::enable_if_t<is_Gaussian_distribution_v<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator*(const S s, Dist&& d)
  {
    auto m = s * mean(d);
    auto c = scale(covariance(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };


  template<
    typename Dist, typename S,
    std::enable_if_t<is_Gaussian_distribution_v<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator/(Dist&& d, const S s)
  {
    auto m = mean(d) / s;
    auto c = inverse_scale(covariance(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };


}

#endif //OPENKALMAN_DISTRIBUTIONTRAITS_H
