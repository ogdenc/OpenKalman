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
#include <iostream>

namespace OpenKalman
{
  template<
    typename Coeffs,
    typename MatrixBase,
    typename CovarianceBase,
    typename random_number_engine = std::mt19937>
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

    /// Construct from a typed matrix (or typed matrix base) and a covariance (or covariance base or symmetric typed matrix).
    template<typename M, typename Cov,
      std::enable_if_t<(is_typed_matrix_v<M> or is_typed_matrix_base_v<M>) and
        (is_covariance_v<Cov> or is_covariance_base_v<Cov> or is_typed_matrix_v<Cov>), int> = 0>
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
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else if constexpr(is_typed_matrix_v<Cov>)
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::RowCoefficients, Coefficients> and
          is_equivalent_v<typename MatrixTraits<Cov>::ColumnCoefficients, Coefficients>);
      else
        static_assert(MatrixTraits<Cov>::dimension == dimension);
    }

    /// Construct from rvalues of a mean and a covariance.
    GaussianDistribution(Mean&& mean, Covariance&& cov) : mu {std::move(mean)}, sigma {std::move(cov)} {}

    /// Construct from an rvalue of a mean and a covariance or covariance base.
    template<typename Cov, std::enable_if_t<is_covariance_v<Cov> or is_covariance_base_v<Cov>, int> = 0>
    GaussianDistribution(Mean&& mean, Cov&& cov) : mu {std::move(mean)}, sigma {std::forward<Cov>(cov)}
    {
      if constexpr(is_covariance_v<Cov>)
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else
        static_assert(MatrixTraits<Cov>::dimension == dimension);
    }

    /// Construct from a mean or typed matrix base and an rvalue of a covariance.
    template<typename M, std::enable_if_t<is_typed_matrix_v<M> or is_typed_matrix_base_v<M>, int> = 0>
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)}
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
    }

    /// Construct using only a covariance, covariance base, or symmetric typed matrix (the mean is set to zero).
    template<typename Cov, std::enable_if_t<is_covariance_v<Cov> or is_covariance_base_v<Cov> or is_typed_matrix_v<Cov>, int> = 0>
    explicit GaussianDistribution(Cov&& cov) : mu {Mean::zero()}, sigma {std::forward<Cov>(cov)}
    {
      if constexpr(is_covariance_v<Cov>)
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else if constexpr(is_typed_matrix_v<Cov>)
        static_assert(is_equivalent_v<typename MatrixTraits<Cov>::RowCoefficients, Coefficients> and
          is_equivalent_v<typename MatrixTraits<Cov>::ColumnCoefficients, Coefficients>);
      else
        static_assert(MatrixTraits<Cov>::dimension == dimension);
    }

    /**********************
     * Assignment Operators
     **********************/

    /// Copy assignment operator.
    GaussianDistribution& operator=(const GaussianDistribution& other)
    {
      mu = other.mu;
      sigma = other.sigma;
      return *this;
    }

    /// Move assignment operator.
    GaussianDistribution& operator=(GaussianDistribution&& other) noexcept
    {
      if (this != &other)
      {
        mu = std::move(other).mu;
        sigma = std::move(other).sigma;
      }
      return *this;
    }

    /// Assign from another compatible distribution.
    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    GaussianDistribution& operator=(Arg&& other) noexcept
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, GaussianDistribution>) if (this == &other) return *this;
      mu = std::forward<Arg>(other).mean();
      sigma = std::forward<Arg>(other).covariance();
      return *this;
    }

  protected:
    template<typename M, typename C>
    auto& increment(M&& m, C&& c)
    {
      mu += std::forward<M>(m);
      sigma += std::forward<C>(c);
      return *this;
    };

  public:
    auto& operator+=(GaussianDistribution&& arg)
    {
      return increment(std::move(arg).mean(), std::move(arg).covariance());
    };

    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    auto& operator+=(Arg&& arg)
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      return increment(std::forward<Arg>(arg).mean(), std::forward<Arg>(arg).covariance());
    };

  protected:
    template<typename M, typename C>
    auto& decrement(M&& m, C&& c)
    {
      mu -= std::forward<M>(m);
      sigma -= std::forward<C>(c);
      return *this;
    };

  public:
    auto& operator-=(GaussianDistribution&& arg)
    {
      return decrement(std::move(arg).mean(), std::move(arg).covariance());
    };

    template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg>, int> = 0>
    auto& operator-=(Arg&& arg)
    {
      static_assert(OpenKalman::is_equivalent_v<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      return decrement(std::forward<Arg>(arg).mean(), std::forward<Arg>(arg).covariance());
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
      auto norm = randomize<TypedMatrix<Coefficients, Axis, MatrixBase>, std::normal_distribution, random_number_engine>(0.0, 1.0);
      auto s = SquareRootCovariance {sigma};
      if constexpr(not is_lower_triangular_v<CovarianceBase>)
        return strict(make_Matrix(mu) + transpose(s) * norm);
      else
        return strict(make_Matrix(mu) + s * norm);
    }

    /// @brief Log-likelihood function for a set of i.i.d. observations z.
    /// @param z One or more i.i.d. observations in the same multivariate space as the mean of the distribution.
    template<typename...Z>
    auto log_likelihood(const Z&...z) const
    {
      static_assert(std::conjunction_v<is_column_vector<Z>...>);
      static_assert(((MatrixTraits<Z>::columns == 1) and ...));
      static_assert(((MatrixTraits<Z>::dimension == dimension) and ...));
      static constexpr auto n = sizeof...(Z);
      static_assert(n >= 1);
      auto sum = (trace(transpose(z - mu) * solve(sigma, z - mu)) + ...);
      return -0.5 * (n * (dimension * std::log(2 * M_PI) + std::log(determinant(sigma))) + sum);
    }

    /// Entropy of the distribution, in bits.
    auto entropy() const
    {
      return 0.5 * (dimension * (1 + std::log2(M_PI) + M_LOG2E) + std::log2(determinant(sigma)));
    }

  protected:
    Mean mu; ///< Mean vector.
    Covariance sigma; ///< Covariance matrix.

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
    std::enable_if_t<is_typed_matrix_v<M> and is_typed_matrix_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<M>::BaseMatrix,
    typename MatrixTraits<typename MatrixTraits<C>::BaseMatrix>::template SelfAdjointBaseType<>>;

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
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0, std::enable_if_t<is_typed_matrix_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::RowCoefficients,
    std::decay_t<M>,
    typename MatrixTraits<typename MatrixTraits<C>::BaseMatrix>::template SelfAdjointBaseType<>>;

  template<typename M, typename C,
    std::enable_if_t<is_typed_matrix_base_v<M>, int> = 0, std::enable_if_t<is_covariance_base_v<C>, int> = 0>
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    Axes<MatrixTraits<M>::dimension>,
    std::decay_t<M>,
    std::decay_t<C>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a Gaussian distribution from another Gaussian distribution.
  template<typename D, std::enable_if_t<is_Gaussian_distribution_v<D>, int> = 0>
  inline auto
  make_GaussianDistribution(D&& dist) noexcept
  {
    return GaussianDistribution(std::forward<D>(dist));
  }


  /// Make a Gaussian distribution from a mean and a covariance.
  template<typename re = std::mt19937, typename M, typename Cov,
    std::enable_if_t<not is_coefficient_v<re> and (is_typed_matrix_v<M> or is_typed_matrix_base_v<M>) and
      (is_covariance_v<Cov> or is_covariance_base_v<Cov> or is_typed_matrix_v<Cov> or is_typed_matrix_base_v<Cov>), int> = 0>
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(is_typed_matrix_v<M>)
    {
      static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
      using C = typename MatrixTraits<M>::RowCoefficients;
      using Mb = typename MatrixTraits<M>::BaseMatrix;

      if constexpr(is_covariance_v<Cov>)
        static_assert(is_equivalent_v<C, typename MatrixTraits<Cov>::Coefficients>);
      else if constexpr(is_typed_matrix_v<Cov>)
        static_assert(is_equivalent_v<C, typename MatrixTraits<Cov>::RowCoefficients> and
          is_equivalent_v<C, typename MatrixTraits<Cov>::ColumnCoefficients>);

      if constexpr(is_covariance_v<Cov> or is_typed_matrix_v<Cov>)
      {
        using Covb = typename MatrixTraits<Cov>::BaseMatrix;
        return GaussianDistribution<C, Mb, Covb, re>(std::forward<M>(mean), std::forward<Cov>(cov));
      }
      else
      {
        auto c = base_matrix(make_Covariance<C>(std::forward<Cov>(cov)));
        return GaussianDistribution<C, Mb, std::decay_t<decltype(c)>, re>(std::forward<M>(mean), std::move(c));
      }
    }
    else if constexpr(is_covariance_v<Cov>)
    {
      using C = typename MatrixTraits<Cov>::Coefficients;
      using Covb = typename MatrixTraits<Cov>::BaseMatrix;
      return GaussianDistribution<C, std::decay_t<M>, Covb, re>(std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else if constexpr(is_typed_matrix_v<Cov>)
    {
      using C = typename MatrixTraits<Cov>::RowCoefficients;
      static_assert(is_equivalent_v<C, typename MatrixTraits<Cov>::ColumnCoefficients>);
      auto sc = base_matrix(make_Covariance(std::forward<Cov>(cov)));
      using SC = std::decay_t<decltype(sc)>;
      return GaussianDistribution<C, std::decay_t<M>, SC, re>(std::forward<M>(mean), std::move(sc));
    }
    else
    {
      using C = Axes<MatrixTraits<M>::dimension>;
      auto c = base_matrix(make_Covariance<C>(std::forward<Cov>(cov)));
      return GaussianDistribution<C, std::decay_t<M>, std::decay_t<decltype(c)>, re>(std::forward<M>(mean), std::move(c));
    }
  }


  /// Make a Gaussian distribution from a typed matrix base and a covariance base or regular matrix for the covariance.
  template<typename Coefficients, typename re = std::mt19937, typename M, typename Cov,
    std::enable_if_t<is_coefficient_v<Coefficients> and
      is_typed_matrix_base_v<M> and (is_covariance_base_v<Cov> or is_typed_matrix_base_v<Cov>), int> = 0>
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(is_covariance_base_v<Cov>)
    {
      return GaussianDistribution<Coefficients, std::decay_t<M>, std::decay_t<Cov>, re>(
        std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else
    {
      auto c = base_matrix(make_Covariance<Coefficients>(std::forward<Cov>(cov)));
      return GaussianDistribution<Coefficients, std::decay_t<M>, std::decay_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /// Make a default Gaussian distribution from a typed matrix (or typed matrix base) and a covariance (or covariance base).
  template<typename M, typename Cov, typename re = std::mt19937,
    std::enable_if_t<(is_typed_matrix_v<M> or is_typed_matrix_base_v<M>) and
      (is_covariance_v<Cov> or is_covariance_base_v<Cov>), int> = 0>
  inline auto
  make_GaussianDistribution()
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(is_typed_matrix_v<M>)
    {
      static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
      using C = typename MatrixTraits<M>::RowCoefficients;
      using Mb = typename MatrixTraits<M>::BaseMatrix;
      if constexpr(is_covariance_v<Cov>) static_assert(is_equivalent_v<C, typename MatrixTraits<Cov>::Coefficients>);
      using Covb = std::conditional_t<is_covariance_v<Cov>, typename MatrixTraits<Cov>::BaseMatrix, std::decay_t<Cov>>;
      return GaussianDistribution<C, Mb, Covb, re>();
    }
    else if constexpr(is_covariance_v<Cov>)
    {
      using C = typename MatrixTraits<Cov>::Coefficients;
      using Covb = typename MatrixTraits<Cov>::BaseMatrix;
      return GaussianDistribution<C, std::decay_t<M>, Covb, re>();
    }
    else
    {
      using C = Axes<MatrixTraits<M>::dimension>;
      return GaussianDistribution<C, std::decay_t<M>, std::decay_t<Cov>, re>();
    }
  }


  /// Make a default Gaussian distribution from a typed matrix base and a covariance base or regular matrix for the covariance.
  template<typename Coefficients, typename M, typename Cov, typename re = std::mt19937,
    std::enable_if_t<is_typed_matrix_base_v<M> and is_covariance_base_v<Cov>, int> = 0>
  inline auto
  make_GaussianDistribution()
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution<Coefficients, std::decay_t<M>, std::decay_t<Cov>, re>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename MatrixBase, typename CovarianceBase, typename re, typename T>
  struct DistributionTraits<GaussianDistribution<Coeffs, MatrixBase, CovarianceBase, re>, T>
  {
    using type = T;
    using Coefficients = Coeffs;
    static constexpr auto dimension = Coefficients::size;
    using Mean = Mean<Coefficients, std::decay_t<MatrixBase>>;
    using Covariance = Covariance<Coefficients, std::decay_t<CovarianceBase>>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;
    template<typename S> using distribution_type = std::normal_distribution<S>;
    using random_number_engine = re;

    template<typename C = Coefficients, typename Mean, typename Covariance,
      std::enable_if_t<(is_typed_matrix_v<Mean> or is_typed_matrix_base_v<Mean>) and
      MatrixTraits<Mean>::columns == 1 and
      (is_covariance_v<Covariance> or is_covariance_base_v<Covariance>), int> = 0>
    static auto make(Mean&& mean, Covariance&& covariance) noexcept
    {
      if constexpr(is_typed_matrix_base_v<Mean> and is_covariance_base_v<Covariance>)
        return make_GaussianDistribution<C, random_number_engine>(
          std::forward<Mean>(mean), std::forward<Covariance>(covariance));
      else
        return make_GaussianDistribution<random_number_engine>(
          std::forward<Mean>(mean), std::forward<Covariance>(covariance));
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


  template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg> and not is_Cholesky_v<Arg>, int> = 0>
  inline auto
  to_Cholesky(Arg&& arg) noexcept
  {
    auto cov = to_Cholesky(covariance(arg));
    return DistributionTraits<Arg>::make(mean(std::forward<Arg>(arg)), std::move(cov));
  }


  template<typename Arg, std::enable_if_t<is_Gaussian_distribution_v<Arg> and is_Cholesky_v<Arg>, int> = 0>
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


  namespace detail
  {
    template<typename Dist, typename Means, typename Covariances, std::size_t ... ints>
    inline auto zip_dist(Means&& ms, Covariances&& cs, std::index_sequence<ints...>)
    {
      return std::tuple {DistributionTraits<Dist>::make(
        std::get<ints>(std::forward<Means>(ms)),
        std::get<ints>(std::forward<Covariances>(cs)))...};
    };
  }


  /// Split distribution.
  template<typename ... Cs, typename D, std::enable_if_t<is_Gaussian_distribution_v<D>, int> = 0>
  inline auto
  split(D&& d) noexcept
  {
    using Coeffs = typename DistributionTraits<D>::Coefficients;
    static_assert(is_prefix_v<Concatenate<Cs...>, Coeffs>);
    if constexpr(sizeof...(Cs) == 1 and is_equivalent_v<Concatenate<Cs...>, Coeffs>)
    {
      return std::tuple(std::forward<D>(d));
    }
    else
    {
      auto means = split<Cs...>(mean(d));
      auto covariances = split<Cs...>(covariance(std::forward<D>(d)));
      return detail::zip_dist<D>(means, covariances, std::make_index_sequence<sizeof...(Cs)>());
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


  ////////////////////////////////
  //    Arithmetic Operators    //
  ////////////////////////////////

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
  operator*(const A& a, const D& d)
  {
    static_assert(not is_Euclidean_transformed_v<A>);
    static_assert(is_equivalent_v<typename MatrixTraits<A>::ColumnCoefficients, typename DistributionTraits<D>::Coefficients>);
    return DistributionTraits<D>::make(a * mean(d), scale(covariance(d), a));
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

#endif //OPENKALMAN_GAUSSIANDISTRIBUTION_H
