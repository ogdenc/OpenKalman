/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * A header file for the class GaussianDistribution and its traits.
 */

#ifndef OPENKALMAN_GAUSSIANDISTRIBUTION_HPP
#define OPENKALMAN_GAUSSIANDISTRIBUTION_HPP

#include <iostream>

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<
    coefficients Coefficients,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine> requires
      (MatrixTraits<MeanNestedMatrix>::dimension == MatrixTraits<CovarianceNestedMatrix>::dimension) and
      (MatrixTraits<MeanNestedMatrix>::columns == 1) and
      (std::is_same_v<typename MatrixTraits<MeanNestedMatrix>::Scalar,
        typename MatrixTraits<CovarianceNestedMatrix>::Scalar>)
#else
  template<
    typename Coefficients,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine>
#endif
  struct GaussianDistribution
  {

    // Redundant in c++20+:
    static_assert(typed_matrix_nestable<MeanNestedMatrix>);
    static_assert(covariance_nestable<CovarianceNestedMatrix>);
    static_assert(MatrixTraits<MeanNestedMatrix>::dimension == MatrixTraits<CovarianceNestedMatrix>::dimension);
    static_assert(MatrixTraits<MeanNestedMatrix>::columns == 1);
    static_assert(std::is_same_v<typename MatrixTraits<MeanNestedMatrix>::Scalar,
      typename MatrixTraits<CovarianceNestedMatrix>::Scalar>);

  protected:

    static constexpr auto dimension = Coefficients::size;
    using Mean = Mean<Coefficients, MeanNestedMatrix>;
    using Covariance = Covariance<Coefficients, CovarianceNestedMatrix>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;

  private:

    template<typename Arg>
    static decltype(auto) cov_adapter(Arg&& arg)
    {
      if constexpr (square_root_covariance<Arg>) return square(std::forward<Arg>(arg));
      else return std::forward<Arg>(arg);
    }

  public:

    /**************
     * Constructors
     **************/

    /*
     * \brief Default constructor.
     */
    GaussianDistribution() : mu {}, sigma {} {}


    /**
     * \brief Copy constructor.
     */
    GaussianDistribution(const GaussianDistribution& other)
      : mu(other.mu), sigma(other.sigma) {}


    /**
     * \brief Move constructor.
     */
    GaussianDistribution(GaussianDistribution&& other) noexcept
      : mu {std::move(other).mu}, sigma {std::move(other).sigma} {}


    /**
     * \brief Construct from related distribution.
     */
#ifdef __cpp_concepts
    template<gaussian_distribution Arg> requires (not std::derived_from<std::decay_t<Arg>, GaussianDistribution>) and
      equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>
#else
    template<typename Arg, std::enable_if_t<gaussian_distribution<Arg> and
      not std::is_base_of_v<GaussianDistribution, std::decay_t<Arg>> and
      equivalent_to<typename DistributionTraits<Arg>::Coefficients, Coefficients>, int> = 0>
#endif
    GaussianDistribution(Arg&& other) noexcept
      : mu {std::forward<Arg>(other).mean_of()}, sigma {std::forward<Arg>(other).covariance_of()} {}


    /**
     * \brief Construct from an rvalue of a mean and a covariance.
     */
    GaussianDistribution(Mean&& mean, Covariance&& cov) : mu {std::move(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from a \ref typed_matrix and an rvalue of a \ref covariance.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable and an rvalue of a \ref covariance.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires column_vector<M> and (MatrixTraits<M>::dimension == dimension)
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and column_vector<M> and
      (MatrixTraits<M>::dimension == dimension), int> = 0>
#endif
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from an rvalue of a \ref mean and a \ref covariance or \ref square_matrix.
     */
#ifdef __cpp_concepts
    template<typename Cov> requires (covariance<Cov> or typed_matrix<Cov>) and
      square_matrix<Cov> and equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>
#else
    template<typename Cov, std::enable_if_t<(covariance<Cov> or typed_matrix<Cov>) and
      square_matrix<Cov> and equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>, int> = 0>
#endif
    GaussianDistribution(Mean&& mean, Cov&& cov) : mu {std::move(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from an rvalue of a \ref mean and a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<covariance_nestable Cov> requires (MatrixTraits<Cov>::dimension == dimension)
#else
    template<typename Cov, std::enable_if_t<covariance_nestable<Cov> and
      (MatrixTraits<Cov>::dimension == dimension), int> = 0>
#endif
    GaussianDistribution(Mean&& mean, Cov&& cov)
      : mu {std::move(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix.
     * \tparam Cov A \ref covariance or \ref square_matrix "square" \ref typed_matrix.
     */
#ifdef __cpp_concepts
    template<typed_matrix M, typename Cov> requires column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov)
      : mu {std::forward<M>(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix.
     * \tparam Cov A \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typed_matrix M, typename Cov> requires column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      covariance_nestable<Cov> and (MatrixTraits<Cov>::dimension == dimension)
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and column_vector<M> and untyped_columns<M> and
      equivalent_to<typename MatrixTraits<M>::RowCoefficients, Coefficients> and
      covariance_nestable<Cov> and (MatrixTraits<Cov>::dimension == dimension), int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix_nestable.
     * \tparam Cov A \ref covariance or \ref square_matrix "square" \ref typed_matrix.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M, typename Cov> requires column_vector<M> and
      (MatrixTraits<M>::dimension == dimension) and (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix_nestable<M> and column_vector<M> and
      (MatrixTraits<M>::dimension == dimension) and (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov)
      : mu {std::forward<M>(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A typed_matrix_nestable.
     * \tparam Cov A \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M, typename Cov> requires column_vector<M> and
      (MatrixTraits<M>::dimension == dimension) and covariance_nestable<Cov> and
      (MatrixTraits<Cov>::dimension == dimension)
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix_nestable<M> and column_vector<M> and
      (MatrixTraits<M>::dimension == dimension) and covariance_nestable<Cov> and
      (MatrixTraits<Cov>::dimension == dimension), int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)} {}


    /// Construct with only a \ref covariance or \ref square_matrix "square" \ref typed_matrix (mean is set to zero).
#ifdef __cpp_concepts
    template<typename Cov> requires (covariance<Cov> or (typed_matrix<Cov> and square_matrix<Cov>)) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>
#else
    template<typename Cov, std::enable_if_t<(covariance<Cov> or (typed_matrix<Cov> and square_matrix<Cov>)) and
      equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, Coefficients>, int> = 0>
#endif
    explicit GaussianDistribution(Cov&& cov) : mu {Mean::zero()}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /// Construct using only a \ref covariance_nestable (the \ref mean is set to zero).
#ifdef __cpp_concepts
    template<covariance_nestable Cov> requires (MatrixTraits<Cov>::dimension == dimension)
#else
    template<typename Cov, std::enable_if_t<
      covariance_nestable<Cov> and (MatrixTraits<Cov>::dimension == dimension), int> = 0>
#endif
    explicit GaussianDistribution(Cov&& cov) : mu {Mean::zero()}, sigma {cov_adapter(std::forward<Cov>(cov))}{}


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

  private:

    template<typename M, typename C>
    static auto
    make(M&& m, C&& c)
    {
      using MB = self_contained_t<nested_matrix_t<M>>;
      using CB = self_contained_t<nested_matrix_t<C>>;
      return GaussianDistribution<Coefficients, MB, CB, random_number_engine>(std::forward<M>(m), std::forward<C>(c));
    }

  public:

    static constexpr auto zero() { return make(Mean::zero(), Covariance::zero()); }

    static constexpr auto normal() { return make(Mean::zero(), Covariance::identity()); }


    auto& mean_of() & { return mu; }

    auto&& mean_of() && { return std::move(mu); }

    const auto& mean_of() const & { return mu; }

    const auto&& mean_of() const && { return std::move(mu); }


    auto& covariance_of() & { return sigma; }

    auto&& covariance_of() && { return std::move(sigma); }

    const auto& covariance_of() const & { return sigma; }

    const auto&& covariance_of() const && { return std::move(sigma); }


    /// \brief Generate a random value from the distribution.
    /// \return A random, single-column typed matrix with probability based on the distribution
    auto operator()() const
    {
      auto norm = randomize<
        Matrix<Coefficients, Axis, MeanNestedMatrix>, std::normal_distribution, random_number_engine>(0.0, 1.0);
      auto s = square_root(sigma);
      if constexpr(not lower_triangular_matrix<CovarianceNestedMatrix>)
        return make_self_contained(make_matrix(mu) + transpose(s) * norm);
      else
        return make_self_contained(make_matrix(mu) + s * norm);
    }


    /// \brief Log-likelihood function for a set of i.i.d. observations z.
    /// \param z One or more i.i.d. observations in the same multivariate space as the mean of the distribution.
    template<typename...Z>
    auto log_likelihood(const Z&...z) const
    {
      static_assert((typed_matrix<Z> and ...));
      static_assert((column_vector<Z> and ...));
      static_assert(((MatrixTraits<Z>::dimension == dimension) and ...));
      static constexpr auto n = sizeof...(Z);
      static_assert(n >= 1);
      auto sum = (trace(transpose(z - mu) * solve(sigma, z - mu)) + ...);
      return -0.5 * (n * (dimension * std::log(2 * std::numbers::pi_v<Scalar>) + std::log(determinant(sigma))) + sum);
    }


    /// Entropy of the distribution, in bits.
    auto entropy() const
    {
      return 0.5 * (dimension * (1 + std::log2(std::numbers::pi_v<Scalar>) + std::numbers::log2e_v<Scalar>)
        + std::log2(determinant(sigma)));
    }

  protected:

    Mean mu; ///< Mean vector.
    Covariance sigma; ///< Covariance matrix.

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<typed_matrix M, covariance C> requires (not square_root_covariance<C>)
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and covariance<C> and not square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    nested_matrix_t<passable_t<M>>,
    nested_matrix_t<passable_t<C>>>;


#ifdef __cpp_concepts
  template<typed_matrix M, square_root_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    nested_matrix_t<passable_t<M>>,
    nested_matrix_t<self_contained_t<decltype(square(std::declval<C>()))>>>;


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
    nested_matrix_t<passable_t<M>>,
    typename MatrixTraits<nested_matrix_t<C>>::template SelfAdjointMatrixFrom<>>;


#ifdef __cpp_concepts
  template<typed_matrix M, covariance_nestable C>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix<M> and covariance_nestable<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<M>::RowCoefficients,
    nested_matrix_t<passable_t<M>>,
    passable_t<C>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, covariance C> requires (not square_root_covariance<C>)
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix_nestable<M> and covariance<C> and not square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::RowCoefficients,
    passable_t<M>,
    nested_matrix_t<passable_t<C>>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, square_root_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix_nestable<M> and square_root_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::RowCoefficients,
    passable_t<M>,
    nested_matrix_t<self_contained_t<decltype(square(std::declval<C&&>()))>>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, typed_matrix C> requires
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix_nestable<M> and typed_matrix<C> and
    equivalent_to<typename MatrixTraits<C>::RowCoefficients, typename MatrixTraits<C>::ColumnCoefficients>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    typename MatrixTraits<C>::RowCoefficients,
    passable_t<M>,
    typename MatrixTraits<nested_matrix_t<C>>::template SelfAdjointMatrixFrom<>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, covariance_nestable C>
#else
  template<typename M, typename C,
    std::enable_if_t<typed_matrix_nestable<M> and covariance_nestable<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    Axes<MatrixTraits<M>::dimension>,
    passable_t<M>,
    passable_t<C>>;


  ///////////////////////////////////
  //        Make functions         //
  ///////////////////////////////////

  /**
   * \brief Make a Gaussian distribution
   * \tparam D Another \ref gaussian_distribution.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<gaussian_distribution D>
#else
  template<typename D, std::enable_if_t<gaussian_distribution<D>, int> = 0>
#endif
  inline auto
  make_GaussianDistribution(D&& dist) noexcept
  {
    return GaussianDistribution {std::forward<D>(dist)};
  }


  /**
   * \brief Make a Gaussian distribution.
   * \tparam re A random number engine.
   * \tparam M A \ref typed_matrix.
   * \tparam Cov A \ref covariance or \ref typed_matrix.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<std::uniform_random_bit_generator re = std::mt19937, typed_matrix M, typename Cov> requires
    column_vector<M> and untyped_columns<M> and square_matrix<Cov> and (covariance<Cov> or typed_matrix<Cov>) and
    (equivalent_to<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Cov>::RowCoefficients>)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<(not coefficients<re>) and
    typed_matrix<M> and column_vector<M> and untyped_columns<M> and
    square_matrix<Cov> and (covariance<Cov> or typed_matrix<Cov>) and
    (equivalent_to<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Cov>::RowCoefficients>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    using C = typename MatrixTraits<M>::RowCoefficients;
    using Mb = passable_t<nested_matrix_t<M>>;
    using Covb = passable_t<nested_matrix_t<Cov>>;
    return GaussianDistribution<C, Mb, Covb, re>(std::forward<M>(mean), std::forward<Cov>(cov));
  }


  /**
   * \brief Make a Gaussian distribution.
   * \tparam re A random number engine.
   * \tparam M A \ref typed_matrix.
   * \tparam Cov A \ref covariance_nestable or \ref typed_matrix_nestable.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<std::uniform_random_bit_generator re = std::mt19937, typed_matrix M, typename Cov> requires
    column_vector<M> and untyped_columns<M> and
    square_matrix<Cov> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    (not coefficients<re>) and typed_matrix<M> and column_vector<M> and untyped_columns<M> and
    square_matrix<Cov> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    using C = typename MatrixTraits<M>::RowCoefficients;
    using Mb = passable_t<nested_matrix_t<M>>;
    auto c = nested_matrix(make_covariance<C>(std::forward<Cov>(cov)));
    return GaussianDistribution<C, Mb, self_contained_t<decltype(c)>, re>(std::forward<M>(mean), std::move(c));
  }


  /**
   * \brief Make a Gaussian distribution.
   * \tparam re A random number engine.
   * \tparam M A \ref typed_matrix_nestable.
   * \tparam Cov A \ref covariance, \ref typed_matrix, \ref covariance_nestable, or \ref typed_matrix_nestable.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<std::uniform_random_bit_generator re = std::mt19937, typed_matrix_nestable M, typename Cov> requires
    column_vector<M> and square_matrix<Cov> and
    (covariance<Cov> or typed_matrix<Cov> or covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    (not coefficients<re>) and typed_matrix_nestable<M> and column_vector<M> and square_matrix<Cov> and
    (covariance<Cov> or typed_matrix<Cov> or covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    if constexpr(covariance<Cov>)
    {
      using C = typename MatrixTraits<Cov>::RowCoefficients;
      using Covb = passable_t<nested_matrix_t<Cov>>;
      return GaussianDistribution<C, passable_t<M>, Covb, re>(std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else if constexpr(typed_matrix<Cov>)
    {
      using C = typename MatrixTraits<Cov>::RowCoefficients;
      auto sc = nested_matrix(make_covariance(std::forward<Cov>(cov)));
      using SC = self_contained_t<decltype(sc)>;
      return GaussianDistribution<C, passable_t<M>, SC, re>(std::forward<M>(mean), std::move(sc));
    }
    else
    {
      static_assert(covariance_nestable<Cov> or typed_matrix_nestable<Cov>);
      using C = Axes<MatrixTraits<M>::dimension>;
      auto c = nested_matrix(make_covariance<C>(std::forward<Cov>(cov)));
      return GaussianDistribution<C, passable_t<M>, self_contained_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /**
   * \brief Make a Gaussian distribution.
   * \tparam Coefficients The types of the \ref coefficients for the distribution.
   * \tparam re A random number engine.
   * \tparam M A \ref typed_matrix_nestable.
   * \tparam Cov A \ref covariance_nestable or \ref typed_matrix_nestable.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, std::uniform_random_bit_generator re = std::mt19937,
    typed_matrix_nestable M, typename Cov> requires
    column_vector<M> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>)
#else
  template<typename Coefficients, typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    coefficients<Coefficients> and typed_matrix_nestable<M> and column_vector<M> and
    (covariance_nestable<Cov> or typed_matrix_nestable<Cov>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension);
    if constexpr(covariance_nestable<Cov>)
    {
      return GaussianDistribution<Coefficients, passable_t<M>, passable_t<Cov>, re>(
        std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else
    {
      static_assert(typed_matrix_nestable<Cov>);
      auto c = nested_matrix(make_covariance<Coefficients>(std::forward<Cov>(cov)));
      return GaussianDistribution<Coefficients, passable_t<M>, self_contained_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /**
   * \brief Make a default Gaussian distribution.
   * \tparam M A \ref typed_matrix.
   * \tparam Cov A \ref covariance.
   * \tparam re A random number engine.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<typed_matrix M, covariance Cov, std::uniform_random_bit_generator re = std::mt19937> requires
    column_vector<M> and untyped_columns<M> and
    equivalent_to<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Cov>::RowCoefficients>
#else
  template<typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    typed_matrix<M> and column_vector<M> and untyped_columns<M> and covariance<Cov> and
    equivalent_to<typename MatrixTraits<M>::RowCoefficients, typename MatrixTraits<Cov>::RowCoefficients>, int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    using C = typename MatrixTraits<M>::RowCoefficients;
    return GaussianDistribution<C, passable_t<nested_matrix_t<M>>, passable_t<nested_matrix_t<Cov>>, re>();
  }


  /**
   * \brief Make a default Gaussian distribution.
   * \tparam M A \ref typed_matrix or \ref typed_matrix_nestable.
   * \tparam Cov A \ref covariance or \ref covariance_nestable.
   * \tparam re A random number engine.
   * \return A \ref gaussian_distribution.
   * \note This overload excludes the case in which M is \ref typed_matrix \em and Cov is \ref covariance.
   */
#ifdef __cpp_concepts
  template<column_vector M, typename Cov, std::uniform_random_bit_generator re = std::mt19937> requires
    (typed_matrix<M> or typed_matrix_nestable<M>) and untyped_columns<M> and
    (covariance<Cov> or covariance_nestable<Cov>) and (not typed_matrix<M> or not covariance<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension)
#else
  template<typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    column_vector<M> and (typed_matrix<M> or typed_matrix_nestable<M>) and untyped_columns<M> and
    (covariance<Cov> or covariance_nestable<Cov>) and (not typed_matrix<M> or not covariance<Cov>) and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension), int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    if constexpr(typed_matrix<M>)
    {
      using C = typename MatrixTraits<M>::RowCoefficients;
      return GaussianDistribution<C, passable_t<nested_matrix_t<M>>, passable_t<Cov>, re>();
    }
    else if constexpr(covariance<Cov>)
    {
      using C = typename MatrixTraits<Cov>::RowCoefficients;
      return GaussianDistribution<C, passable_t<M>, passable_t<nested_matrix_t<Cov>>, re>();
    }
    else
    {
      using C = Axes<MatrixTraits<M>::dimension>;
      return GaussianDistribution<C, passable_t<M>, passable_t<Cov>, re>();
    }
  }


  /**
   * \brief Make a default Gaussian distribution.
   * \tparam Coefficients The types of the \ref coefficients for the distribution.
   * \tparam M A \ref typed_matrix_nestable
   * \tparam Cov A \ref covariance_nestable.
   * \tparam re A random number engine
   * \return A \ref gaussian_distribution
   */
#ifdef __cpp_concepts
  template<coefficients Coefficients, typed_matrix_nestable M, covariance_nestable Cov,
      std::uniform_random_bit_generator re = std::mt19937> requires
    column_vector<M> and (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension)
#else
  template<typename Coefficients, typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    typed_matrix_nestable<M> and column_vector<M> and covariance_nestable<Cov> and
    (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension), int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    return GaussianDistribution<Coefficients, passable_t<M>, passable_t<Cov>, re>();
  }


  ///////////////////////////
  //        Traits         //
  ///////////////////////////

  template<typename Coeffs, typename NestedMean, typename NestedCovariance, typename re>
  struct DistributionTraits<GaussianDistribution<Coeffs, NestedMean, NestedCovariance, re>>
  {
    using Coefficients = Coeffs;
    static constexpr auto dimension = Coefficients::size;
    using Mean = Mean<Coefficients, NestedMean>;
    using Covariance = Covariance<Coefficients, NestedCovariance>;
    using Scalar = typename MatrixTraits<Mean>::Scalar;
    template<typename S> using distribution_type = std::normal_distribution<S>;
    using random_number_engine = re;

    using SelfContainedFrom = GaussianDistribution<Coefficients, self_contained_t<NestedMean>,
      self_contained_t<NestedCovariance>>;


#ifdef __cpp_concepts
    template<typename C = Coefficients, typed_matrix_nestable M, covariance_nestable Cov> requires
      column_vector<M> and (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension)
#else
    template<typename C = Coefficients, typename M, typename Cov,
      std::enable_if_t<typed_matrix_nestable<M> and covariance_nestable<Cov> and
      column_vector<M> and (MatrixTraits<M>::dimension == MatrixTraits<Cov>::dimension), int> = 0>
#endif
    static auto make(M&& mean, Cov&& covariance) noexcept
    {
      return make_GaussianDistribution<C, random_number_engine>(
        std::forward<M>(mean), std::forward<Cov>(covariance));
    }

    static auto zero() { return make(MatrixTraits<NestedMean>::zero(), MatrixTraits<NestedCovariance>::zero()); }

    static auto normal() { return make(MatrixTraits<NestedMean>::zero(), MatrixTraits<NestedCovariance>::identity()); }
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


  /// Convert to self-contained version of the distribution.
  template<typename...Ts, typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr(
      (self_contained<typename DistributionTraits<Arg>::Mean> and
        self_contained<typename DistributionTraits<Arg>::Covariance>) or
      ((sizeof...(Ts) > 0) and ... and std::is_lvalue_reference_v<Ts>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return make_GaussianDistribution(make_self_contained(mean_of(arg)), make_self_contained(covariance_of(arg)));
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
      return make_GaussianDistribution(std::move(m), std::move(cov));
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
      return std::tuple {make_GaussianDistribution(
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
    return make_GaussianDistribution(std::move(m1), std::move(m2));
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
    return make_GaussianDistribution(std::move(m1), std::move(m2));
  };


#ifdef __cpp_concepts
  template<typed_matrix A, gaussian_distribution D> requires gaussian_distribution<D> and
    (not euclidean_transformed<A>) and
    (equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, typename DistributionTraits<D>::Coefficients>)
#else
  template<typename A, typename D, std::enable_if_t<typed_matrix<A> and gaussian_distribution<D> and
    (not euclidean_transformed<A>) and
    (equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, typename DistributionTraits<D>::Coefficients>), int> = 0>
#endif
  inline auto
  operator*(A&& a, D&& d)
  {
    auto m = make_self_contained(a * mean_of(d));
    auto c = make_self_contained(scale(covariance_of(std::forward<D>(d)), std::forward<A>(a)));
    return make_GaussianDistribution(std::move(m), std::move(c));
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
    return make_GaussianDistribution(std::move(m), std::move(c));
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
    return make_GaussianDistribution(std::move(m), std::move(c));
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
    return make_GaussianDistribution(std::move(m), std::move(c));
  };

}

#endif //OPENKALMAN_GAUSSIANDISTRIBUTION_HPP
