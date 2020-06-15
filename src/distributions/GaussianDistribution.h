/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_GAUSSIANDISTRIBUTION_H
#define OPENKALMAN_GAUSSIANDISTRIBUTION_H

#include <random>
#include "variables/classes/Mean.h"
#include "variables/classes/TypedMatrix.h"
#include "variables/support/TypedMatrixOverloads.h"
#include "variables/classes/Covariance.h"
#include "variables/classes/SquareRootCovariance.h"
#include "variables/support/CovarianceOverloads.h"

namespace OpenKalman
{
  template<
    typename Coeffs,
    typename MatrixBase,
    typename CovarianceBase>
  struct GaussianDistribution
  {
    using Coefficients = Coeffs;
    using Mean = Mean<Coefficients, MatrixBase>;
    using Covariance = Covariance<Coefficients, CovarianceBase>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;
    static constexpr auto dimension = Coefficients::size;
    static_assert(is_typed_matrix_base_v<MatrixBase>);
    static_assert(is_covariance_base_v<CovarianceBase>);
    static_assert(MatrixTraits<MatrixBase>::dimension == MatrixTraits<CovarianceBase>::dimension);
    static_assert(MatrixTraits<MatrixBase>::columns == 1);
    static_assert(std::is_same_v<Scalar, typename MatrixTraits<Covariance>::Scalar>);



  protected:
    template<typename Arg1, typename Arg2>
    static auto
    make(Arg1&& arg_mean, Arg2&& arg_moment)
    {
      return DistributionTraits<GaussianDistribution>::make(std::forward<Arg1>(arg_mean), std::forward<Arg2>(arg_moment));
    }

  public:
    /**************
     * Constructors
     **************/

    /// Default constructor.
    GaussianDistribution() : mu {}, sigma {} {}

    /// Copy constructor.
    GaussianDistribution(const GaussianDistribution& other)
      : mu(other.mu), sigma(other.sigma) {}

    /// Move constructor.
    GaussianDistribution(GaussianDistribution&& other) noexcept
      : mu {std::move(other).mu}, sigma {std::move(other).sigma} {}

    /// Construct from related distribution.
    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    GaussianDistribution(Arg&& other) noexcept
      : mu {std::forward<Arg>(other).mean()}, sigma {std::forward<Arg>(other).covariance()}
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
    }

    /// Construct from a typed matrix (or typed matrix base) and a covariance (or covariance base).
    template<typename M, typename Cov,
      std::enable_if_t<(is_typed_matrix_v<M> or is_typed_matrix_base_v<M>) and
        (is_covariance_v<Cov> or is_covariance_base_v<Cov>), int> = 0>
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)}
    {
      static_assert(MatrixTraits<M>::columns == 1);
      if constexpr(is_typed_matrix_v<M>)
      {
        static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
        static_assert(is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      }
      else
      {
        static_assert(MatrixTraits<M>::dimension == dimension);
      }
      if constexpr(is_covariance_v<Cov>)
      {
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      }
      else
      {
        static_assert(MatrixTraits<Cov>::dimension == dimension);
      }
    }

    /// Construct using only a covariance or covariance base (the mean is set to zero).
    template<typename Cov, std::enable_if_t<is_covariance_v<Cov> or is_covariance_base_v<Cov>, int> = 0>
    explicit GaussianDistribution(Cov&& cov) : mu {Mean::zero()}, sigma {std::forward<Cov>(cov)}
    {
      if constexpr(is_covariance_v<Cov>)
      {
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      }
      else
      {
        static_assert(MatrixTraits<Cov>::dimension == dimension);
      }
    }

    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    GaussianDistribution& operator=(const GaussianDistribution& other)
    {
      mu = other.x;
      sigma = other.sigma;
      return *this;
    }

    /// Move assignment operator.
    GaussianDistribution& operator=(GaussianDistribution&& other) noexcept
    {
      if (this != &other)
      {
        mu = std::move(other).x;
        sigma = std::move(other).sigma;
        return *this;
      }
    }

    /// Assign from another compatible distribution.
    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    GaussianDistribution& operator=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      mu = std::forward<Arg>(other).mean();
      sigma = std::forward<Arg>(other).covariance();
      return *this;
    }

    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    auto& operator+=(Arg&& v)
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      mu += std::forward<Arg>(v).mean();
      sigma += std::forward<Arg>(v).covariance();
      return *this;
    };

    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    auto& operator-=(Arg&& v)
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      mu -= std::forward<Arg>(v).mean();
      sigma += std::forward<Arg>(v).covariance();
      return *this;
    };

    template<typename S, std::enable_if_t<std::is_convertible_v<S, const Scalar>, int> = 0>
    auto& operator*=(const S scale)
    {
      mu *= static_cast<const Scalar>(scale);
      sigma.scale(static_cast<const Scalar>(scale));
      return *this;
    };

    template<typename S, std::enable_if_t<std::is_convertible_v<S, const Scalar>, int> = 0>
    auto& operator/=(const S scale)
    {
      mu /= static_cast<const Scalar>(scale);
      sigma.inverse_scale(static_cast<const Scalar>(scale));
      return *this;
    };

    /*********
     * Other
     *********/

    static constexpr auto zero() { return make(Mean::zero(), Covariance::zero()); }

    static constexpr auto normal() { return make(Mean::zero(), Covariance::identity()); }

    constexpr auto& mean() & { return mu; }

    constexpr auto&& mean() && { return std::move(mu); }

    constexpr const auto& mean() const & { return mu; }

    constexpr const auto&& mean() const && { return std::move(mu); }

    constexpr auto& covariance() & { return sigma; }

    constexpr auto&& covariance() && { return std::move(sigma); }

    constexpr const auto& covariance() const & { return sigma; }

    constexpr const auto&& covariance() const && { return std::move(sigma); }

    /// @brief Generate a random value from the distribution.
    /// @return A random, single-column typed matrix with probability based on the distribution
    auto operator()() const
    {
      auto norm = randomize<TypedMatrix<Coefficients, Axis, MatrixBase>>();
      auto s = SquareRootCovariance {sigma};
      if constexpr(not is_lower_triangular_v<CovarianceBase>)
        return strict(make_Matrix(mu) + transpose(s) * norm);
      else
        return strict(make_Matrix(mu) + s * norm);
    }

    /// @brief Log likelihood function for a single observation z.
    /// @param z Location in the same multivariate space as the mean.
    template<typename Z>
    auto log_probability(Z&& z) const
    {
      static_assert(is_column_vector_v<Z>);
      static_assert(MatrixTraits<Z>::columns == 1);
      static_assert(MatrixTraits<Z>::dimension == dimension);
      const auto diff = std::forward<Z>(z) - mu;
      return -0.5 * dimension * std::log(2 * M_PI) - std::log(determinant(sigma)) - transpose(diff) * solve(sigma, diff);
    }

    template<typename Z>
    auto entropy() const
    {
      return -std::log(determinant(2 * M_PI * sigma)) - dimension / 2;
    }

  protected:
    Mean mu; ///< Mean vector.
    Covariance sigma; ///< Covariance matrix.

    template<typename, typename, typename> friend
    struct GaussianDistribution;

    template<
      template<typename, typename> typename,
      typename,
      typename,
      typename,
      typename ...>
    friend
    struct KalmanFilter;

    template<
      typename DY, typename PxyType, typename Z,
      std::enable_if_t<is_covariance_v<typename DistributionTraits<DY>::Covariance> and not is_Cholesky_v<DY>, int> = 0,
      std::enable_if_t<is_typed_matrix_v<PxyType>, int> = 0,
      std::enable_if_t<is_mean_v<Z>, int> = 0>
    auto&
    kalmanUpdate(DY&& Dist_y, PxyType&& P_xy, Z&& z)
    {
      static_assert(MatrixTraits<Z>::columns == 1);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::ColumnCoefficients, typename DistributionTraits<DY>::Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Z>::RowCoefficients, typename DistributionTraits<DY>::Coefficients>);
      // Effectively, P_xx -= P_xy * adjoint(inverse(P_yy)) * adjoint(P_xy)
      using CoeffsM = typename DistributionTraits<DY>::Coefficients;
      auto y = mean(std::forward<DY>(Dist_y));
      auto P_yy = covariance(std::forward<DY>(Dist_y));
      auto K = make_Matrix<Coefficients, CoeffsM>(adjoint(solve(adjoint(std::move(P_yy)), adjoint(P_xy))));
      // Note: K == P_xy * inverse(P_yy)
      mu += K * (std::forward<Z>(z) - std::move(y)); // == K(z - y)
      sigma -= std::move(P_xy) * adjoint(std::move(K)); // == K * P_yy * adjoint(K)
      return *this;
    }

    /// @TODO Need to adapt this
    template<
      typename DY, typename PxyType, typename Z,
      std::enable_if_t<is_covariance_v<typename DistributionTraits<DY>::Covariance> and is_Cholesky_v<DY>, int> = 0,
      std::enable_if_t<is_typed_matrix_v<PxyType>, int> = 0,
      std::enable_if_t<is_mean_v<Z>, int> = 0>
    auto&
    kalmanUpdate(DY&& Dist_y, PxyType&& P_xy, Z&& z)
    {
      static_assert(MatrixTraits<Z>::columns == 1);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::ColumnCoefficients, typename DistributionTraits<DY>::Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Z>::RowCoefficients, typename DistributionTraits<DY>::Coefficients>);
      using CoeffsM = typename DistributionTraits<DY>::Coefficients;
      auto y = mean(std::forward<DY>(Dist_y));
      auto S_yy = sqrt_covariance(std::forward<DY>(Dist_y));
      auto K = make_Matrix<Coefficients, CoeffsM>(adjoint(solve(adjoint(S_yy), adjoint(std::forward<PxyType>(P_xy)))));
      mu += K * (std::forward<Z>(z) - std::move(y));
      sigma -= std::move(K) * std::move(S_yy);
      // Note: this is equivalent to P_xx -= K * P_yy * K.adjoint() == K * S_yy * (K * S_yy).adjoint();
      return *this;
    }

  };


  /////////////////////////////////////
  //        Deduction Guides         //
  /////////////////////////////////////

  template<typename D, std::enable_if_t<is_Gaussian_distribution_v<D>, int> = 0>
  GaussianDistribution(D&&) -> GaussianDistribution<
    typename DistributionTraits<D>::Coefficients,
    typename MatrixTraits<typename DistributionTraits<D>::Mean>::BaseMatrix,
    typename MatrixTraits<typename DistributionTraits<D>::Covariance>::BaseMatrix>;

  template<typename M, typename C,
    std::enable_if_t<is_typed_matrix_v<M> and is_covariance_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<M>::BaseMatrix,
    typename MatrixTraits<C>::BaseMatrix>;

  template<typename M, typename C,
    std::enable_if_t<is_typed_matrix_v<M> and is_covariance_base_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
  typename MatrixTraits<M>::RowCoefficients,
  typename MatrixTraits<M>::BaseMatrix,
  std::decay_t<C>>;

  template<typename M, typename C,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0, std::enable_if_t<is_covariance_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::Coefficients,
    std::decay_t<M>,
    typename MatrixTraits<C>::BaseMatrix>;

  template<typename M, typename C,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0, std::enable_if_t<is_covariance_base_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    Axes<MatrixTraits<M>::dimension>,
    std::decay_t<M>,
    std::decay_t<C>>;


  //////////////////////////////////
  //        Make Function         //
  //////////////////////////////////

  template<typename M, typename Cov,
    std::enable_if_t<is_column_vector_v<M>, int> = 0,
    std::enable_if_t<is_covariance_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<Cov>::Coefficients>);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution(std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<typename M, typename Cov,
    std::enable_if_t<is_column_vector_v<M>, int> = 0,
    std::enable_if_t<is_typed_matrix_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Cov>::RowCoefficients>);
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Cov>::RowCoefficients, typename MatrixTraits<Cov>::ColumnCoefficients>);
    return GaussianDistribution(std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<typename M, typename Cov,
    std::enable_if_t<is_column_vector_v<M> and is_covariance_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution(std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<
    typename M, typename Cov,
    std::enable_if_t<is_column_vector_v<M>, int> = 0,
    std::enable_if_t<is_typed_matrix_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    static_assert(MatrixTraits<Cov>::dimension == MatrixTraits<Cov>::columns);
    using Coefficients = typename MatrixTraits<M>::RowCoefficients;
    using SAType = typename MatrixTraits<Cov>::template SelfAdjointBaseType<>;
    return make_GaussianDistribution(std::forward<M>(mean), MatrixTraits<SAType>::make(std::forward<Cov>(covariance)));
  }

  template<typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_covariance_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution(std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_typed_matrix_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Cov>::RowCoefficients, typename MatrixTraits<Cov>::ColumnCoefficients>);
    return GaussianDistribution(std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<
    typename Coefficients, typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_covariance_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution<Coefficients, std::decay_t<M>, std::decay_t<Cov>>(
      std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<
    typename Coefficients, typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_typed_matrix_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    static_assert(MatrixTraits<Cov>::dimension == MatrixTraits<Cov>::columns);
    using SAType = typename MatrixTraits<Cov>::template SelfAdjointBaseType<>;
    return GaussianDistribution<Coefficients, std::decay_t<M>, SAType>(
      std::forward<M>(mean), MatrixTraits<SAType>::make(std::forward<Cov>(covariance)));
  }

  template<
    typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_covariance_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    using Coefficients = Axes<MatrixTraits<M>::dimension>;
    return GaussianDistribution<Coefficients, std::decay_t<M>, std::decay_t<Cov>>(
      std::forward<M>(mean), std::forward<Cov>(covariance));
  }

  template<
    typename M, typename Cov,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0,
    std::enable_if_t<is_typed_matrix_base_v<Cov>, int> = 0>
  inline auto make_GaussianDistribution(M&& mean, Cov&& covariance) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    static_assert(MatrixTraits<Cov>::dimension == MatrixTraits<Cov>::columns);
    using Coefficients = Axes<MatrixTraits<M>::dimension>;
    using SAType = typename MatrixTraits<Cov>::template SelfAdjointBaseType<>;
    return GaussianDistribution<Coefficients, std::decay_t<M>, SAType>(
      std::forward<M>(mean), MatrixTraits<SAType>::make(std::forward<Cov>(covariance)));
  }


}

#endif //OPENKALMAN_GAUSSIANDISTRIBUTION_H
