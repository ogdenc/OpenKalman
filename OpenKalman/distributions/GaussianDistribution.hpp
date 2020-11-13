/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * @file GaussianDistribution.h
 * A header file for the class GaussianDistribution and its traits.
 */

#ifndef OPENKALMAN_GAUSSIANDISTRIBUTION_HPP
#define OPENKALMAN_GAUSSIANDISTRIBUTION_HPP

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
    static_assert(self_contained<MatrixBase>, "MatrixBase must be self-contained.");
    static_assert(self_contained<CovarianceBase>, "CovarianceBase must be self-contained.");
    static_assert(typed_matrix_base<MatrixBase>);
    static_assert(covariance_base<CovarianceBase>);
    static_assert(MatrixTraits<MatrixBase>::dimension == MatrixTraits<CovarianceBase>::dimension);
    static_assert(MatrixTraits<MatrixBase>::columns == 1);
    static_assert(std::is_same_v<Scalar, typename MatrixTraits<Covariance>::Scalar>);

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
    template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
    GaussianDistribution(Arg&& other) noexcept
      : mu {std::forward<Arg>(other).mean_of()}, sigma {std::forward<Arg>(other).covariance_of()}
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
    }

    /// Construct from an rvalue of a mean and a covariance.
    GaussianDistribution(Mean&& mean, Covariance&& cov) : mu {std::move(mean)}, sigma {std::move(cov)} {}

    /// Construct from a typed matrix (or typed matrix base) and an rvalue of a covariance.
#ifdef __cpp_concepts
    template<typename M> requires typed_matrix<M> or typed_matrix_base<M>
#else
    template<typename M, std::enable_if_t<typed_matrix<M> or typed_matrix_base<M>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)}
    {
      static_assert(MatrixTraits<M>::columns == 1);
      if constexpr(typed_matrix<M>)
      {
        static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
        static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      }
      else
      {
        static_assert(MatrixTraits<M>::dimension == dimension);
      }
    }

    /// Construct from an rvalue of a mean and a covariance (or covariance base or symmetric typed matrix).
#ifdef __cpp_concepts
    template<typename Cov> requires covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>
#else
    template<typename Cov, std::enable_if_t<
      covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>, int> = 0>
#endif
    GaussianDistribution(Mean&& mean, Cov&& cov) : mu {std::move(mean)}, sigma {std::forward<Cov>(cov)}
    {
      if constexpr(covariance<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else if constexpr(typed_matrix<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients> and
          equivalent_to<typename MatrixTraits<Cov>::ColumnCoefficients, Coefficients>);
      else
        static_assert(MatrixTraits<Cov>::dimension == dimension);
    }

    /// Construct from a typed matrix (or typed matrix base) and a covariance (or covariance base or symmetric typed matrix).
#ifdef __cpp_concepts
    template<typename M, typename Cov> requires (typed_matrix<M> or typed_matrix_base<M>) and
      (covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>)
#else
    template<typename M, typename Cov, std::enable_if_t<(typed_matrix<M> or typed_matrix_base<M>) and
      (covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>), int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)}
    {
      static_assert(MatrixTraits<M>::columns == 1);
      if constexpr(typed_matrix<M>)
      {
        static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
        static_assert(equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>);
      }
      else
      {
        static_assert(MatrixTraits<M>::dimension == dimension);
      }
      if constexpr(covariance<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else if constexpr(typed_matrix<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients> and
          equivalent_to<typename MatrixTraits<Cov>::ColumnCoefficients, Coefficients>);
      else
        static_assert(MatrixTraits<Cov>::dimension == dimension);
    }

    /// Construct using only a covariance, covariance base, or symmetric typed matrix (the mean is set to zero).
#ifdef __cpp_concepts
    template<typename Cov> requires covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>
#else
    template<typename Cov, std::enable_if_t<
      covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov>, int> = 0>
#endif
    explicit GaussianDistribution(Cov&& cov) : mu {Mean::zero()}, sigma {std::forward<Cov>(cov)}
    {
      if constexpr(covariance<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::Coefficients, Coefficients>);
      else if constexpr(typed_matrix<Cov>)
        static_assert(equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients> and
          equivalent_to<typename MatrixTraits<Cov>::ColumnCoefficients, Coefficients>);
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
    template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
    GaussianDistribution& operator=(Arg&& other) noexcept
    {
      static_assert(equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>);
      if constexpr (std::is_same_v<std::decay_t<Arg>, GaussianDistribution>) if (this == &other) return *this;
      mu = std::forward<Arg>(other).mean_of();
      sigma = std::forward<Arg>(other).covariance_of();
      return *this;
    }

    auto& operator+=(const GaussianDistribution& arg)
    {
      mu += arg.mean_of();
      sigma += arg.covariance_of();
      return *this;
    };

    auto& operator-=(const GaussianDistribution& arg)
    {
      mu -= arg.mean_of();
      sigma -= arg.covariance_of();
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

  protected:
    template<typename M, typename C>
    static auto
    make(M&& m, C&& c)
    {
      using MB = self_contained_t<typename MatrixTraits<M>::BaseMatrix>;
      using CB = self_contained_t<typename MatrixTraits<C>::BaseMatrix>;
      return GaussianDistribution<Coefficients, MB, CB, random_number_engine>(std::forward<M>(m), std::forward<C>(c));
    }

  public:
    static constexpr auto zero() { return make(Mean::zero(), Covariance::zero()); }

    static constexpr auto normal() { return make(Mean::zero(), Covariance::identity()); }

    constexpr auto& mean_of() & { return mu; }

    constexpr auto&& mean_of() && { return std::move(mu); }

    constexpr const auto& mean_of() const & { return mu; }

    constexpr const auto&& mean_of() const && { return std::move(mu); }

    constexpr auto& covariance_of() & { return sigma; }

    constexpr auto&& covariance_of() && { return std::move(sigma); }

    constexpr const auto& covariance_of() const & { return sigma; }

    constexpr const auto&& covariance_of() const && { return std::move(sigma); }

    /// @brief Generate a random value from the distribution.
    /// @return A random, single-column typed matrix with probability based on the distribution
    auto operator()() const
    {
      auto norm = randomize<
        Matrix<Coefficients, Axis, MatrixBase>, std::normal_distribution, random_number_engine>(0.0, 1.0);
      auto s = square_root(sigma);
      if constexpr(not lower_triangular_matrix<CovarianceBase>)
        return make_self_contained(make_Matrix(mu) + transpose(s) * norm);
      else
        return make_self_contained(make_Matrix(mu) + s * norm);
    }

    /// @brief Log-likelihood function for a set of i.i.d. observations z.
    /// @param z One or more i.i.d. observations in the same multivariate space as the mean of the distribution.
    template<typename...Z>
    auto log_likelihood(const Z&...z) const
    {
      static_assert((column_vector<Z> and ...));
      static_assert(((MatrixTraits<Z>::columns == 1) and ...));
      static_assert(((MatrixTraits<Z>::dimension == dimension) and ...));
      static constexpr auto n = sizeof...(Z);
      static_assert(n >= 1);
      auto sum = (trace(transpose(z - mu) * solve(sigma, z - mu)) + ...);
      return -0.5 * (n * (dimension * std::log(2 * std::numbers::pi_v<Scalar>) + std::log(determinant(sigma))) + sum);
    }

    /// Entropy of the distribution, in bits.
    auto entropy() const
    {
      return 0.5 * (dimension * (1 + std::log2(std::numbers::pi_v<Scalar>) + M_LOG2E) + std::log2(determinant(sigma)));
    }

  protected:
    Mean mu; ///< Mean vector.
    Covariance sigma; ///< Covariance matrix.

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<gaussian_distribution D>
#else
  template<typename D, std::enable_if_t<gaussian_distribution<D>, int> = 0>
#endif
  GaussianDistribution(D&&) -> GaussianDistribution<
    typename DistributionTraits<D>::Coefficients,
    typename MatrixTraits<typename DistributionTraits<passable_t<D>>::Mean>::BaseMatrix,
    typename MatrixTraits<typename DistributionTraits<passable_t<D>>::Covariance>::BaseMatrix>;


#ifdef __cpp_concepts
  template<typed_matrix M, covariance C> requires (not square_root_covariance<C>)
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and covariance<C> and not square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<passable_t<M>>::BaseMatrix,
    typename MatrixTraits<passable_t<C>>::BaseMatrix>;


#ifdef __cpp_concepts
  template<typed_matrix M, square_root_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<passable_t<M>>::BaseMatrix,
    typename MatrixTraits<self_contained_t<decltype(square(std::declval<C>()))>>::BaseMatrix>;


#ifdef __cpp_concepts
  template<typed_matrix M, typed_matrix C> requires
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and typed_matrix<C> and
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<passable_t<M>>::BaseMatrix,
    typename MatrixTraits<typename MatrixTraits<C>::BaseMatrix>::template SelfAdjointBaseType<>>;


#ifdef __cpp_concepts
  template<typed_matrix M, covariance_base C>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix<M> and covariance_base<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    typename MatrixTraits<passable_t<M>>::BaseMatrix,
    passable_t<C>>;


#ifdef __cpp_concepts
  template<typed_matrix_base M, covariance C> requires (not square_root_covariance<C>)
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix_base<M> and covariance<C> and not square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::Coefficients,
    passable_t<M>,
    typename MatrixTraits<passable_t<C>>::BaseMatrix>;


#ifdef __cpp_concepts
  template<typed_matrix_base M, square_root_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix_base<M> and square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::Coefficients,
    passable_t<M>,
    typename MatrixTraits<self_contained_t<decltype(square(std::declval<C&&>()))>>::BaseMatrix>;


#ifdef __cpp_concepts
  template<typed_matrix_base M, typed_matrix C> requires
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix_base<M> and typed_matrix<C> and
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::RowCoefficients,
    passable_t<M>,
    typename MatrixTraits<typename MatrixTraits<C>::BaseMatrix>::template SelfAdjointBaseType<>>;


#ifdef __cpp_concepts
  template<typed_matrix_base M, covariance_base C>
#else
  template<typename M, typename C,
    std::enable_if_t<typed_matrix_base<M> and covariance_base<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    Axes<MatrixTraits<M>::dimension>,
    passable_t<M>,
    passable_t<C>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /// Make a Gaussian distribution from another Gaussian distribution.
#ifdef __cpp_concepts
  template<typename D> requires gaussian_distribution<D>
#else
  template<typename D, std::enable_if_t<gaussian_distribution<D>, int> = 0>
#endif
  inline auto
  make_GaussianDistribution(D&& dist) noexcept
  {
    return GaussianDistribution {std::forward<D>(dist)};
  }


  /// Make a Gaussian distribution from a mean and a covariance.
#ifdef __cpp_concepts
  template<typename re = std::mt19937, typename M, typename Cov> requires
    (not coefficients<re>) and (typed_matrix<M> or typed_matrix_base<M>) and
    (covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov> or typed_matrix_base<Cov>)
#else
  template<typename re = std::mt19937, typename M, typename Cov,
    std::enable_if_t<not is_coefficients_v<re> and (typed_matrix<M> or typed_matrix_base<M>) and
      (covariance<Cov> or covariance_base<Cov> or typed_matrix<Cov> or typed_matrix_base<Cov>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(typed_matrix<M>)
    {
      static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
      using C = typename MatrixTraits<M>::RowCoefficients;
      using Mb = typename MatrixTraits<M>::BaseMatrix;

      if constexpr(covariance<Cov>)
        static_assert(equivalent_to<C, typename MatrixTraits<Cov>::Coefficients>);
      else if constexpr(typed_matrix<Cov>)
        static_assert(equivalent_to<C, typename MatrixTraits<Cov>::RowCoefficients> and
          equivalent_to<C, typename MatrixTraits<Cov>::ColumnCoefficients>);

      if constexpr(covariance<Cov> or typed_matrix<Cov>)
      {
        using Covb = typename MatrixTraits<Cov>::BaseMatrix;
        return GaussianDistribution<C, passable_t<Mb>, passable_t<Covb>, re>(std::forward<M>(mean), std::forward<Cov>(cov));
      }
      else
      {
        auto c = base_matrix(make_Covariance<C>(std::forward<Cov>(cov)));
        return GaussianDistribution<C, passable_t<Mb>, self_contained_t<decltype(c)>, re>(std::forward<M>(mean), std::move(c));
      }
    }
    else if constexpr(covariance<Cov>)
    {
      using C = typename MatrixTraits<Cov>::Coefficients;
      using Covb = typename MatrixTraits<Cov>::BaseMatrix;
      return GaussianDistribution<C, passable_t<M>, passable_t<Covb>, re>(std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else if constexpr(typed_matrix<Cov>)
    {
      using C = typename MatrixTraits<Cov>::RowCoefficients;
      static_assert(equivalent_to<C, typename MatrixTraits<Cov>::ColumnCoefficients>);
      auto sc = base_matrix(make_Covariance(std::forward<Cov>(cov)));
      using SC = self_contained_t<decltype(sc)>;
      return GaussianDistribution<C, passable_t<M>, SC, re>(std::forward<M>(mean), std::move(sc));
    }
    else
    {
      using C = Axes<MatrixTraits<M>::dimension>;
      auto c = base_matrix(make_Covariance<C>(std::forward<Cov>(cov)));
      return GaussianDistribution<C, passable_t<M>, self_contained_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /// Make a Gaussian distribution from a typed matrix base and a covariance base or regular matrix for the covariance.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typename re = std::mt19937, typed_matrix_base M, typename Cov> requires
    covariance_base<Cov> or typed_matrix_base<Cov>
#else
  template<typename Coefficients, typename re = std::mt19937, typename M, typename Cov,
    std::enable_if_t<is_coefficients_v<Coefficients> and
      typed_matrix_base<M> and (covariance_base<Cov> or typed_matrix_base<Cov>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(covariance_base<Cov>)
    {
      return GaussianDistribution<Coefficients, passable_t<M>, passable_t<Cov>, re>(
        std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else
    {
      auto c = base_matrix(make_Covariance<Coefficients>(std::forward<Cov>(cov)));
      return GaussianDistribution<Coefficients, passable_t<M>, self_contained_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /// Make a default Gaussian distribution from a typed matrix (or typed matrix base) and a covariance (or covariance base).
#ifdef __cpp_concepts
  template<typename M, typename Cov, typename re = std::mt19937> requires
    (typed_matrix<M> or typed_matrix_base<M>) and (covariance<Cov> or covariance_base<Cov>)
#else
  template<typename M, typename Cov, typename re = std::mt19937,
    std::enable_if_t<(typed_matrix<M> or typed_matrix_base<M>) and
      (covariance<Cov> or covariance_base<Cov>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(typed_matrix<M>)
    {
      static_assert(MatrixTraits<M>::ColumnCoefficients::axes_only);
      using C = typename MatrixTraits<M>::RowCoefficients;
      using Mb = typename MatrixTraits<M>::BaseMatrix;
      if constexpr(covariance<Cov>) static_assert(equivalent_to<C, typename MatrixTraits<Cov>::Coefficients>);
      using Covb = std::conditional_t<covariance<Cov>, typename MatrixTraits<Cov>::BaseMatrix, std::decay_t<Cov>>;
      return GaussianDistribution<C, passable_t<Mb>, passable_t<Covb>, re>();
    }
    else if constexpr(covariance<Cov>)
    {
      using C = typename MatrixTraits<Cov>::Coefficients;
      using Covb = typename MatrixTraits<Cov>::BaseMatrix;
      return GaussianDistribution<C, passable_t<M>, passable_t<Covb>, re>();
    }
    else
    {
      using C = Axes<MatrixTraits<M>::dimension>;
      return GaussianDistribution<C, passable_t<M>, passable_t<Cov>, re>();
    }
  }


  /// Make a default Gaussian distribution from a typed matrix base and a covariance base or regular matrix for the covariance.
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_base M, covariance_base Cov, typename re = std::mt19937>
#else
  template<typename Coefficients, typename M, typename Cov, typename re = std::mt19937,
    std::enable_if_t<typed_matrix_base<M> and covariance_base<Cov>, int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    static_assert(MatrixTraits<M>::columns == 1);
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    return GaussianDistribution<Coefficients, passable_t<M>, passable_t<Cov>, re>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename MatrixBase, typename CovarianceBase, typename re>
  struct DistributionTraits<GaussianDistribution<Coeffs, MatrixBase, CovarianceBase, re>>
  {
    using Coefficients = Coeffs;
    static constexpr auto dimension = Coefficients::size;
    using Mean = Mean<Coefficients, MatrixBase>;
    using Covariance = Covariance<Coefficients, CovarianceBase>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;
    template<typename S> using distribution_type = std::normal_distribution<S>;
    using random_number_engine = re;

    using SelfContained = GaussianDistribution<Coefficients, typename MatrixTraits<MatrixBase>::SelfContained,
      typename MatrixTraits<CovarianceBase>::SelfContained>;

#ifdef __cpp_concepts
    template<typename C = Coefficients, typename Mean, typename Covariance> requires
      (typed_matrix<Mean> or typed_matrix_base<Mean>) and (MatrixTraits<Mean>::columns == 1) and
      (covariance<Covariance> or covariance_base<Covariance>)
#else
    template<typename C = Coefficients, typename Mean, typename Covariance,
      std::enable_if_t<(typed_matrix<Mean> or typed_matrix_base<Mean>) and
      MatrixTraits<Mean>::columns == 1 and
      (covariance<Covariance> or covariance_base<Covariance>), int> = 0>
#endif
    static auto make(Mean&& mean, Covariance&& covariance) noexcept
    {
      if constexpr(typed_matrix_base<Mean> and covariance_base<Covariance>)
        return make_GaussianDistribution<C, random_number_engine>(
          std::forward<Mean>(mean), std::forward<Covariance>(covariance));
      else
        return make_GaussianDistribution<random_number_engine>(
          std::forward<Mean>(mean), std::forward<Covariance>(covariance));
    }

    static auto zero() { return make(MatrixTraits<Mean>::zero(), MatrixTraits<Covariance>::zero()); }

    static auto normal() { return make(MatrixTraits<Mean>::zero(), MatrixTraits<Covariance>::identity()); }
  };


  //////////////////////////////
  //        Overloads         //
  //////////////////////////////

  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
  constexpr decltype(auto)
  mean_of(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).mean_of();
  }


  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
  constexpr decltype(auto)
  covariance_of(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).covariance_of();
  }


  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg> and not cholesky_form<Arg> and
    not diagonal_matrix<Arg>, int> = 0>
  inline auto
  to_Cholesky(Arg&& arg) noexcept
  {
    auto cov = to_Cholesky(covariance_of(arg));
    return DistributionTraits<Arg>::make(mean_of(std::forward<Arg>(arg)), std::move(cov));
  }


  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg> and cholesky_form<Arg> and
    not diagonal_matrix<Arg>, int> = 0>
  inline auto
  from_Cholesky(Arg&& arg) noexcept
  {
    auto cov = from_Cholesky(covariance_of(arg));
    return DistributionTraits<Arg>::make(mean_of(std::forward<Arg>(arg)), std::move(cov));
  }


  /// Convert to self-contained version of the distribution.
  template<typename...Ts, typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(self_contained<Arg> or (std::is_lvalue_reference_v<Ts> and ... and (sizeof...(Ts) > 0)))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return DistributionTraits<Arg>::make(make_self_contained(mean_of(arg)), make_self_contained(covariance_of(arg)));
    }
  }


  template<typename D, typename ... Ds,
    std::enable_if_t<(gaussian_distribution<D> and ... and gaussian_distribution<Ds>), int> = 0>
  auto
  concatenate(const D& d, const Ds& ... ds)
  {
    if constexpr(sizeof...(Ds) > 0)
    {
      auto m = concatenate(mean_of(d), mean_of(ds)...);
      auto cov = concatenate(covariance_of(d), covariance_of(ds)...);
      return DistributionTraits<D>::template make(std::move(m), std::move(cov));
    }
    else
    {
      return d;
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
  template<typename ... Cs, typename D, std::enable_if_t<gaussian_distribution<D>, int> = 0>
  inline auto
  split(D&& d) noexcept
  {
    using Coeffs = typename DistributionTraits<D>::Coefficients;
    static_assert(prefix_of<Concatenate<Cs...>, Coeffs>);
    if constexpr(sizeof...(Cs) == 1 and equivalent_to<Concatenate<Cs...>, Coeffs>)
    {
      return std::tuple(std::forward<D>(d));
    }
    else
    {
      auto means = split_vertical<Cs...>(mean_of(d));
      auto covariances = split_diagonal<Cs...>(covariance_of(std::forward<D>(d)));
      return detail::zip_dist<D>(means, covariances, std::make_index_sequence<sizeof...(Cs)>());
    }
  }


  template<typename Dist, std::enable_if_t<gaussian_distribution<Dist>, int> = 0>
  inline std::ostream&
  operator<<(std::ostream& os, const Dist& d)
  {
    os << "mean:" << std::endl << mean_of(d) << std::endl <<
    "covariance:" << std::endl << covariance_of(d) << std::endl;
    return os;
  }


  ////////////////////////////////
  //    Arithmetic Operators    //
  ////////////////////////////////

  template<
    typename Dist1,
    typename Dist2,
    std::enable_if_t<gaussian_distribution<Dist1> and gaussian_distribution<Dist2> and
      equivalent_to<typename DistributionTraits<Dist1>::Coefficients,
        typename DistributionTraits<Dist2>::Coefficients>, int> = 0>
  inline auto
  operator+(const Dist1& d1, const Dist2& d2)
  {
    auto m1 = mean_of(d1) + mean_of(d2);
    auto m2 = covariance_of(d1) + covariance_of(d2);
    return DistributionTraits<Dist1>::make(std::move(m1), std::move(m2));
  };


  template<
    typename Dist1,
    typename Dist2,
    std::enable_if_t<gaussian_distribution<Dist1> and gaussian_distribution<Dist2> and
      equivalent_to<typename DistributionTraits<Dist1>::Coefficients,
        typename DistributionTraits<Dist2>::Coefficients>, int> = 0>
  inline auto
  operator-(const Dist1& d1, const Dist2& d2)
  {
    auto m1 = mean_of(d1) - mean_of(d2);
    auto m2 = covariance_of(d1) - covariance_of(d2);
    return DistributionTraits<Dist1>::make(std::move(m1), std::move(m2));
  };


#ifdef __cpp_concepts
  template<typed_matrix A, typename D> requires gaussian_distribution<D>
#else
  template<typename A, typename D, std::enable_if_t<
    typed_matrix<A> and gaussian_distribution<D>, int> = 0>
#endif
  inline auto
  operator*(const A& a, const D& d)
  {
    static_assert(not euclidean_transformed<A>);
    static_assert(equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, typename DistributionTraits<D>::Coefficients>);
    return DistributionTraits<D>::make(a * mean_of(d), scale(covariance_of(d), a));
  }


  template<
    typename Dist, typename S,
    std::enable_if_t<gaussian_distribution<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator*(Dist&& d, const S s)
  {
    auto m = mean_of(d) * s;
    auto c = scale(covariance_of(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };


  template<
    typename Dist, typename S,
    std::enable_if_t<gaussian_distribution<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator*(const S s, Dist&& d)
  {
    auto m = s * mean_of(d);
    auto c = scale(covariance_of(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };


  template<
    typename Dist, typename S,
    std::enable_if_t<gaussian_distribution<Dist> and
      std::is_convertible_v<S, typename DistributionTraits<Dist>::Scalar>, int> = 0>
  inline auto
  operator/(Dist&& d, const S s)
  {
    auto m = mean_of(d) / s;
    auto c = inverse_scale(covariance_of(d), s);
    return DistributionTraits<Dist>::make(std::move(m), std::move(c));
  };

}

#endif //OPENKALMAN_GAUSSIANDISTRIBUTION_HPP
