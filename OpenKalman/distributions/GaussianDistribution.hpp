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
#include <special-matrices/details/special-matrix-interface.hpp>


namespace OpenKalman
{
#ifdef __cpp_concepts
  template<
    fixed_vector_space_descriptor FixedDescriptor,
    typed_matrix_nestable MeanNestedMatrix,
    covariance_nestable CovarianceNestedMatrix,
    std::uniform_random_bit_generator random_number_engine> requires
      (index_dimension_of_v<MeanNestedMatrix, 0> == index_dimension_of_v<CovarianceNestedMatrix, 0>) and
      (index_dimension_of_v<MeanNestedMatrix, 1> == 1) and
      (std::is_same_v<scalar_type_of_t<MeanNestedMatrix>,
        scalar_type_of_t<CovarianceNestedMatrix>>)
#else
  template<
    typename FixedDescriptor,
    typename MeanNestedMatrix,
    typename CovarianceNestedMatrix,
    typename random_number_engine>
#endif
  struct GaussianDistribution
  {

#ifndef __cpp_concepts
    static_assert(typed_matrix_nestable<MeanNestedMatrix>);
    static_assert(covariance_nestable<CovarianceNestedMatrix>);
    static_assert(index_dimension_of_v<MeanNestedMatrix, 0> == index_dimension_of_v<CovarianceNestedMatrix, 0>);
    static_assert(index_dimension_of_v<MeanNestedMatrix, 1> == 1);
    static_assert(std::is_same_v<scalar_type_of_t<MeanNestedMatrix>,
      scalar_type_of_t<CovarianceNestedMatrix>>);
#endif

  protected:

    static constexpr auto dim = dimension_size_of_v<FixedDescriptor>;
    using Mean = Mean<FixedDescriptor, MeanNestedMatrix>;
    using Covariance = Covariance<FixedDescriptor, CovarianceNestedMatrix>;
    using Scalar = scalar_type_of_t<Mean>;

  private:

    template<typename Arg>
    static decltype(auto) cov_adapter(Arg&& arg)
    {
      if constexpr (triangular_covariance<Arg>) return square(std::forward<Arg>(arg));
      else return std::forward<Arg>(arg);
    }

  public:

    // -------------- //
    //  Constructors  //
    // -------------- //

    /*
     * \brief Default constructor.
     */
#ifdef __cpp_concepts
    GaussianDistribution() requires std::default_initializable<Mean> and std::default_initializable<Covariance>
#else
    template<typename T = Mean, typename U = Covariance, std::enable_if_t<
      std::is_default_constructible_v<T> and std::is_default_constructible_v<U>, int> = 0>
    GaussianDistribution()
#endif
      : mu {}, sigma {} {}


    /**
     * \brief Construct from related distribution.
     */
#ifdef __cpp_concepts
    template<gaussian_distribution Arg> requires (not std::derived_from<std::decay_t<Arg>, GaussianDistribution>) and
      equivalent_to<typename DistributionTraits<Arg>::FixedDescriptor, FixedDescriptor>
#else
    template<typename Arg, std::enable_if_t<gaussian_distribution<Arg> and
      not std::is_base_of_v<GaussianDistribution, std::decay_t<Arg>> and
      equivalent_to<typename DistributionTraits<Arg>::FixedDescriptor, FixedDescriptor>, int> = 0>
#endif
    GaussianDistribution(Arg&& arg) noexcept
      : mu {std::forward<Arg>(arg).mu}, sigma {std::forward<Arg>(arg).sigma} {}


    /**
     * \brief Construct from an rvalue of a mean and a covariance.
     */
    GaussianDistribution(Mean&& mean, Covariance&& cov) : mu {std::move(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from a \ref typed_matrix and an rvalue of a \ref covariance.
     */
#ifdef __cpp_concepts
    template<typed_matrix M> requires vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor>
#else
    template<typename M, std::enable_if_t<typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from a \ref typed_matrix_nestable and an rvalue of a \ref covariance.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M> requires vector<M> and (index_dimension_of_v<M, 0> == dim)
#else
    template<typename M, std::enable_if_t<typed_matrix_nestable<M> and vector<M> and
      (index_dimension_of<M, 0>::value == dim), int> = 0>
#endif
    GaussianDistribution(M&& mean, Covariance&& cov) : mu {std::forward<M>(mean)}, sigma {std::move(cov)} {}


    /**
     * \brief Construct from an rvalue of a \ref mean and a \ref covariance, \ref typed_matrix,
     * or \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Cov> requires ((covariance<Cov> or (typed_matrix<Cov> and square_shaped<Cov>)) and
        equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>) or
      (covariance_nestable<Cov> and index_dimension_of_v<Cov, 0> == dim)
#else
    template<typename Cov, std::enable_if_t<(covariance<Cov> or (typed_matrix<Cov> and square_shaped<Cov>)) and
        equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>, int> = 0>
    GaussianDistribution(Mean&& mean, Cov&& cov) : mu {std::move(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}

    template<typename Cov, std::enable_if_t<
      covariance_nestable<Cov> and (index_dimension_of<Cov, 0>::value == dim), int> = 0>
#endif
    GaussianDistribution(Mean&& mean, Cov&& cov) : mu {std::move(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix.
     * \tparam Cov A \ref covariance or \ref square_shaped "square" \ref typed_matrix.
     */
#ifdef __cpp_concepts
    template<typed_matrix M, typename Cov> requires vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor> and
      (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor> and
      (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov)
      : mu {std::forward<M>(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix.
     * \tparam Cov A \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typed_matrix M, typename Cov> requires vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor> and
      covariance_nestable<Cov> and (index_dimension_of_v<Cov, 0> == dim)
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and
      equivalent_to<vector_space_descriptor_of_t<M, 0>, FixedDescriptor> and
      covariance_nestable<Cov> and (index_dimension_of<Cov, 0>::value == dim), int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A \ref typed_matrix_nestable.
     * \tparam Cov A \ref covariance or \ref square_shaped "square" \ref typed_matrix.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M, typename Cov> requires vector<M> and
      (index_dimension_of_v<M, 0> == dim) and (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix_nestable<M> and vector<M> and
      (index_dimension_of<M, 0>::value == dim) and (covariance<Cov> or typed_matrix<Cov>) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>, int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov)
      : mu {std::forward<M>(mean)}, sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /**
     * \brief Construct from matrices representing a mean and a covariance.
     * \tparam M A typed_matrix_nestable.
     * \tparam Cov A \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typed_matrix_nestable M, typename Cov> requires vector<M> and
      (index_dimension_of_v<M, 0> == dim) and covariance_nestable<Cov> and
      (index_dimension_of_v<Cov, 0> == dim)
#else
    template<typename M, typename Cov, std::enable_if_t<typed_matrix_nestable<M> and vector<M> and
      (index_dimension_of<M, 0>::value == dim) and covariance_nestable<Cov> and
      (index_dimension_of<Cov, 0>::value == dim), int> = 0>
#endif
    GaussianDistribution(M&& mean, Cov&& cov) : mu {std::forward<M>(mean)}, sigma {std::forward<Cov>(cov)} {}


    /// Construct with only a \ref covariance or \ref square_shaped "square" \ref typed_matrix (mean is set to zero).
#ifdef __cpp_concepts
    template<typename Cov> requires (covariance<Cov> or (typed_matrix<Cov> and square_shaped<Cov>)) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>
#else
    template<typename Cov, std::enable_if_t<(covariance<Cov> or (typed_matrix<Cov> and square_shaped<Cov>)) and
      equivalent_to<vector_space_descriptor_of_t<Cov, 0>, FixedDescriptor>, int> = 0>
#endif
    explicit GaussianDistribution(Cov&& cov) :
      mu {[]{ make_zero<MeanNestedMatrix>(get_vector_space_descriptor<0>(cov), Dimensions<1>{}); }()},
      sigma {cov_adapter(std::forward<Cov>(cov))} {}


    /// Construct using only a \ref covariance_nestable (the \ref mean is set to zero).
#ifdef __cpp_concepts
    template<covariance_nestable Cov> requires (index_dimension_of_v<Cov, 0> == dim)
#else
    template<typename Cov, std::enable_if_t<
      covariance_nestable<Cov> and (index_dimension_of<Cov, 0>::value == dim), int> = 0>
#endif
    explicit GaussianDistribution(Cov&& cov) :
      mu {[]{ make_zero<MeanNestedMatrix>(get_vector_space_descriptor<0>(cov), Dimensions<1>{}); }()},
      sigma {cov_adapter(std::forward<Cov>(cov))}{}

    // ---------------------- //
    //  Assignment Operators  //
    // ---------------------- //

    /// Assign from another compatible distribution.
#ifdef __cpp_concepts
    template<gaussian_distribution Arg> requires
      equivalent_to<typename DistributionTraits<Arg>::FixedDescriptor, FixedDescriptor>
#else
    template<typename Arg, std::enable_if_t<gaussian_distribution<Arg> and
      equivalent_to<typename DistributionTraits<Arg>::FixedDescriptor, FixedDescriptor>, int> = 0>
#endif
    GaussianDistribution& operator=(Arg&& arg) noexcept
    {
      if constexpr (std::is_same_v<std::decay_t<Arg>, GaussianDistribution>) if (this == &arg) return *this;
      mu = std::forward<Arg>(arg).mu;
      sigma = std::forward<Arg>(arg).sigma;
      return *this;
    }


    auto& operator+=(const GaussianDistribution& arg)
    {
      mu += arg.mu;
      sigma += arg.sigma;
      return *this;
    };


    auto& operator-=(const GaussianDistribution& arg)
    {
      mu -= arg.mu;
      sigma -= arg.sigma;
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

    // ------- //
    //  Other  //
    // ------- //

  private:

    template<typename M, typename C>
    static auto
    make(M&& m, C&& c)
    {
      using MB = equivalent_self_contained_t<nested_object_of_t<M>>;
      using CB = equivalent_self_contained_t<nested_object_of_t<C>>;
      return GaussianDistribution<FixedDescriptor, MB, CB, random_number_engine>(std::forward<M>(m), std::forward<C>(c));
    }

  public:

    /**
     * \brief Generate a random value from the distribution.
     * \return A random, single-column typed matrix with probability based on the distribution
     */
    auto operator()() const
    {
      auto norm = randomize<Matrix<FixedDescriptor, Axis, MeanNestedMatrix>, random_number_engine>(
        std::normal_distribution {0.0, 1.0});
      auto s = square_root(sigma);
      if constexpr(triangular_matrix<decltype(s), TriangleType::upper>)
        return make_self_contained(Matrix {mu} + transpose(std::move(s)) * std::move(norm));
      else
        return make_self_contained(Matrix {mu} + std::move(s) * std::move(norm));
    }


    /**
     * \brief \brief Log-likelihood function for a set of i.i.d. observations z.
     * \param z One or more i.i.d. observations in the same multivariate space as the mean of the distribution.
     */
#ifdef __cpp_concepts
    template<typed_matrix...Z> requires (sizeof...(Z) > 0) and
      ((vector<Z> and equivalent_to<vector_space_descriptor_of_t<Z, 0>, FixedDescriptor>) and ...)
#else
    template<typename...Z, std::enable_if_t<(sizeof...(Z) > 0) and ((typed_matrix<Z> and vector<Z> and
      equivalent_to<vector_space_descriptor_of_t<Z, 0>, FixedDescriptor>) and ...), int> = 0>
#endif
    Scalar log_likelihood(const Z&...z) const
    {
      static constexpr auto n = sizeof...(Z);
      auto sum = (trace(transpose(z - mu) * solve(sigma, z - mu)) + ...);
      return -0.5 * (n * (dim * interface::scalar_traits<Scalar>::log(2 * numbers::pi_v<long double>) +
        interface::scalar_traits<Scalar>::log(determinant(sigma))) + sum);
    }


    /// Entropy of the distribution, in bits.
    Scalar entropy() const
    {
      return 0.5 * (dim * (1 + interface::scalar_traits<Scalar>::log2(numbers::pi_v<long double>) +
        Scalar {numbers::log2e_v<long double>}) + interface::scalar_traits<Scalar>::log2(determinant(sigma)));
    }


    /**
     * \brief The mean of the distribution.
     */
    Mean mu;


    /**
     * \brief The Covariance matrix of the distribution.
     */
    Covariance sigma;

  };


  // ------------------------------- //
  //        Deduction Guides         //
  // ------------------------------- //

#ifdef __cpp_concepts
  template<typed_matrix M, self_adjoint_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix<M> and self_adjoint_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<M, 0>,
    nested_object_of_t<passable_t<M>>,
    nested_object_of_t<passable_t<C>>>;


#ifdef __cpp_concepts
  template<typed_matrix M, triangular_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and triangular_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<M, 0>,
    nested_object_of_t<passable_t<M>>,
    nested_object_of_t<equivalent_self_contained_t<decltype(square(std::declval<C>()))>>>;


#ifdef __cpp_concepts
  template<typed_matrix M, typed_matrix C> requires
    equivalent_to<vector_space_descriptor_of_t<C, 0>, vector_space_descriptor_of_t<C, 1>>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix<M> and typed_matrix<C> and
    equivalent_to<vector_space_descriptor_of_t<C, 0>, vector_space_descriptor_of_t<C, 1>>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<M, 0>,
    nested_object_of_t<passable_t<M>>,
    decltype(to_covariance_nestable(nested_object(std::declval<passable_t<C>>())))>;


#ifdef __cpp_concepts
  template<typed_matrix M, covariance_nestable C>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix<M> and covariance_nestable<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<M, 0>,
    nested_object_of_t<passable_t<M>>,
    passable_t<C>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, self_adjoint_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix_nestable<M> and self_adjoint_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<C, 0>,
    passable_t<M>,
    nested_object_of_t<passable_t<C>>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, triangular_covariance C>
#else
  template<typename M, typename C, std::enable_if_t<
    typed_matrix_nestable<M> and triangular_covariance<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<C, 0>,
    passable_t<M>,
    nested_object_of_t<equivalent_self_contained_t<decltype(square(std::declval<C&&>()))>>>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, typed_matrix C> requires
    equivalent_to<vector_space_descriptor_of_t<C, 0>, vector_space_descriptor_of_t<C, 1>>
#else
  template<typename M, typename C, std::enable_if_t<typed_matrix_nestable<M> and typed_matrix<C> and
    equivalent_to<vector_space_descriptor_of_t<C, 0>, vector_space_descriptor_of_t<C, 1>>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    vector_space_descriptor_of_t<C, 0>,
    passable_t<M>,
    decltype(to_covariance_nestable(nested_object(std::declval<passable_t<C>>())))>;


#ifdef __cpp_concepts
  template<typed_matrix_nestable M, covariance_nestable C>
#else
  template<typename M, typename C,
    std::enable_if_t<typed_matrix_nestable<M> and covariance_nestable<C>, int> = 0>
#endif
  GaussianDistribution(M&&, C&&) -> GaussianDistribution<
    Dimensions<index_dimension_of_v<M, 0>>,
    passable_t<M>,
    passable_t<C>>;


  // ----------------------------- //
  //        Make functions         //
  // ----------------------------- //

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
    vector<M> and has_untyped_index<M, 1> and square_shaped<Cov> and (covariance<Cov> or typed_matrix<Cov>) and
    (equivalent_to<vector_space_descriptor_of_t<M, 0>, vector_space_descriptor_of_t<Cov, 0>>)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<(not fixed_vector_space_descriptor<re>) and
    typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and
    square_shaped<Cov> and (covariance<Cov> or typed_matrix<Cov>) and
    (equivalent_to<vector_space_descriptor_of_t<M, 0>, vector_space_descriptor_of_t<Cov, 0>>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    using C = vector_space_descriptor_of_t<M, 0>;
    using Mb = passable_t<nested_object_of_t<M>>;
    using Covb = passable_t<nested_object_of_t<Cov>>;
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
    vector<M> and has_untyped_index<M, 1> and
    square_shaped<Cov> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    (not fixed_vector_space_descriptor<re>) and typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and
    square_shaped<Cov> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (index_dimension_of<M, 0>::value == index_dimension_of<Cov, 0>::value), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    using C = vector_space_descriptor_of_t<M, 0>;
    using Mb = passable_t<nested_object_of_t<M>>;
    auto c = nested_object(make_covariance<C>(std::forward<Cov>(cov)));
    return GaussianDistribution<C, Mb, equivalent_self_contained_t<decltype(c)>, re>(std::forward<M>(mean), std::move(c));
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
    vector<M> and square_shaped<Cov> and
    (covariance<Cov> or typed_matrix<Cov> or covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>)
#else
  template<typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    (not fixed_vector_space_descriptor<re>) and typed_matrix_nestable<M> and vector<M> and square_shaped<Cov> and
    (covariance<Cov> or typed_matrix<Cov> or covariance_nestable<Cov> or typed_matrix_nestable<Cov>) and
    (index_dimension_of<M, 0>::value == index_dimension_of<Cov, 0>::value), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    if constexpr(covariance<Cov>)
    {
      using C = vector_space_descriptor_of_t<Cov, 0>;
      using Covb = passable_t<nested_object_of_t<Cov>>;
      return GaussianDistribution<C, passable_t<M>, Covb, re>(std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else if constexpr(typed_matrix<Cov>)
    {
      using C = vector_space_descriptor_of_t<Cov, 0>;
      auto sc = nested_object(make_covariance(std::forward<Cov>(cov)));
      using SC = equivalent_self_contained_t<decltype(sc)>;
      return GaussianDistribution<C, passable_t<M>, SC, re>(std::forward<M>(mean), std::move(sc));
    }
    else
    {
      static_assert(covariance_nestable<Cov> or typed_matrix_nestable<Cov>);
      using C = Dimensions<index_dimension_of_v<M, 0>>;
      auto c = nested_object(make_covariance<C>(std::forward<Cov>(cov)));
      return GaussianDistribution<C, passable_t<M>, equivalent_self_contained_t<decltype(c)>, re>(
        std::forward<M>(mean), std::move(c));
    }
  }


  /**
   * \brief Make a Gaussian distribution.
   * \tparam FixedDescriptor The types of the \ref fixed_vector_space_descriptor for the distribution.
   * \tparam re A random number engine.
   * \tparam M A \ref typed_matrix_nestable.
   * \tparam Cov A \ref covariance_nestable or \ref typed_matrix_nestable.
   * \return A \ref gaussian_distribution.
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, std::uniform_random_bit_generator re = std::mt19937,
    typed_matrix_nestable M, typename Cov> requires vector<M> and (covariance_nestable<Cov> or typed_matrix_nestable<Cov>)
#else
  template<typename FixedDescriptor, typename re = std::mt19937, typename M, typename Cov, std::enable_if_t<
    fixed_vector_space_descriptor<FixedDescriptor> and typed_matrix_nestable<M> and vector<M> and
    (covariance_nestable<Cov> or typed_matrix_nestable<Cov>), int> = 0>
#endif
  inline auto
  make_GaussianDistribution(M&& mean, Cov&& cov) noexcept
  {
    static_assert(index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>);
    if constexpr(covariance_nestable<Cov>)
    {
      return GaussianDistribution<FixedDescriptor, passable_t<M>, passable_t<Cov>, re>(
        std::forward<M>(mean), std::forward<Cov>(cov));
    }
    else
    {
      static_assert(typed_matrix_nestable<Cov>);
      auto c = nested_object(make_covariance<FixedDescriptor>(std::forward<Cov>(cov)));
      return GaussianDistribution<FixedDescriptor, passable_t<M>, equivalent_self_contained_t<decltype(c)>, re>(
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
  vector<M> and has_untyped_index<M, 1> and
    equivalent_to<vector_space_descriptor_of_t<M, 0>, vector_space_descriptor_of_t<Cov, 0>>
#else
  template<typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    typed_matrix<M> and vector<M> and has_untyped_index<M, 1> and covariance<Cov> and
    equivalent_to<vector_space_descriptor_of_t<M, 0>, vector_space_descriptor_of_t<Cov, 0>>, int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    using C = vector_space_descriptor_of_t<M, 0>;
    return GaussianDistribution<C, passable_t<nested_object_of_t<M>>, passable_t<nested_object_of_t<Cov>>, re>();
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
  template<vector M, typename Cov, std::uniform_random_bit_generator re = std::mt19937> requires
    (typed_matrix<M> or typed_matrix_nestable<M>) and has_untyped_index<M, 1> and
    (covariance<Cov> or covariance_nestable<Cov>) and (not typed_matrix<M> or not covariance<Cov>) and
    (index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>)
#else
  template<typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    vector<M> and (typed_matrix<M> or typed_matrix_nestable<M>) and has_untyped_index<M, 1> and
    (covariance<Cov> or covariance_nestable<Cov>) and (not typed_matrix<M> or not covariance<Cov>) and
    (index_dimension_of<M, 0>::value == index_dimension_of<Cov, 0>::value), int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    if constexpr(typed_matrix<M>)
    {
      using C = vector_space_descriptor_of_t<M, 0>;
      return GaussianDistribution<C, passable_t<nested_object_of_t<M>>, passable_t<Cov>, re>();
    }
    else if constexpr(covariance<Cov>)
    {
      using C = vector_space_descriptor_of_t<Cov, 0>;
      return GaussianDistribution<C, passable_t<M>, passable_t<nested_object_of_t<Cov>>, re>();
    }
    else
    {
      using C = Dimensions<index_dimension_of_v<M, 0>>;
      return GaussianDistribution<C, passable_t<M>, passable_t<Cov>, re>();
    }
  }


  /**
   * \brief Make a default Gaussian distribution.
   * \tparam FixedDescriptor The types of the \ref fixed_vector_space_descriptor for the distribution.
   * \tparam M A \ref typed_matrix_nestable
   * \tparam Cov A \ref covariance_nestable.
   * \tparam re A random number engine
   * \return A \ref gaussian_distribution
   */
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor FixedDescriptor, typed_matrix_nestable M, covariance_nestable Cov,
      std::uniform_random_bit_generator re = std::mt19937> requires
  vector<M> and (index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>)
#else
  template<typename FixedDescriptor, typename M, typename Cov, typename re = std::mt19937, std::enable_if_t<
    typed_matrix_nestable<M> and vector<M> and covariance_nestable<Cov> and
    (index_dimension_of<M, 0>::value == index_dimension_of<Cov, 0>::value), int> = 0>
#endif
  inline auto
  make_GaussianDistribution()
  {
    return GaussianDistribution<FixedDescriptor, passable_t<M>, passable_t<Cov>, re>();
  }


  // --------------------- //
  //        Traits         //
  // --------------------- //

  namespace interface
  {
    template<typename Coeffs, typename NestedMean, typename NestedCovariance, typename re>
    struct indexible_object_traits<GaussianDistribution<Coeffs, NestedMean, NestedCovariance, re>>
    {
      using scalar_type = scalar_type_of_t<NestedMean>;

      template<typename Arg>
      static constexpr auto count_indices(const Arg& arg) { return std::integral_constant<std::size_t, 1>{}; }

      template<typename Arg, typename N>
      static constexpr auto get_vector_space_descriptor(const Arg& arg, N n)
      {
        if constexpr (static_index_value<N>)
        {
          static_assert(n == 0_uz);
          if constexpr (not dynamic_dimension<NestedMean, 0>) return OpenKalman::get_vector_space_descriptor<0>(mean_of(arg));
          else return OpenKalman::get_vector_space_descriptor<0>(covariance_of(arg));
        }
        else
        {
          return OpenKalman::get_vector_space_descriptor(mean_of(arg), n);
        }
      }

      using dependents = std::tuple<NestedMean, NestedCovariance>;

      static constexpr bool has_runtime_parameters = false;

      template<typename Arg>
      static decltype(auto) nested_object(Arg&& arg)
      {
        return std::forward<Arg>(arg).mu;
      }

      template<typename Arg>
      static auto convert_to_self_contained(Arg&& arg)
      {
        auto&& m = make_self_contained(std::forward<Arg>(arg).mu);
        using M = decltype(m);
        auto&& s = make_self_contained(std::forward<Arg>(arg).sigma);
        using S = decltype(s);
        return GaussianDistribution<Coeffs, M, S, re> {std::forward<M>(m), std::forward<M>(s)};
      }


#ifdef __cpp_lib_concepts
      template<typename Arg> requires raw_data_defined_for<Mean<Coeffs, NestedMean>>
#else
      template<typename Arg, std::enable_if_t<raw_data_defined_for<Mean<Coeffs, NestedMean>>, int> = 0>
#endif
      static constexpr auto *
      raw_data(Arg& arg) { return internal::raw_data(arg.mu()); }


      static constexpr Layout layout = layout_of_v<NestedMean>;

    };

  } // namespace interface


  template<typename Coeffs, typename NestedMean, typename NestedCovariance, typename re>
  struct DistributionTraits<GaussianDistribution<Coeffs, NestedMean, NestedCovariance, re>>
  {
    using FixedDescriptor = Coeffs;
    using Mean = Mean<FixedDescriptor, NestedMean>;
    using Covariance = Covariance<FixedDescriptor, NestedCovariance>;
    using Scalar = scalar_type_of_t<Mean>;
    template<typename S> using distribution_type = std::normal_distribution<S>;
    using random_number_engine = re;

#ifdef __cpp_concepts
    template<fixed_vector_space_descriptor C = FixedDescriptor, typed_matrix_nestable M, covariance_nestable Cov> requires
      vector<M> and (index_dimension_of_v<M, 0> == index_dimension_of_v<Cov, 0>)
#else
    template<typename C = FixedDescriptor, typename M, typename Cov,
      std::enable_if_t<fixed_vector_space_descriptor<C> and typed_matrix_nestable<M> and covariance_nestable<Cov> and
      vector<M> and (index_dimension_of<M, 0>::value == index_dimension_of<Cov, 0>::value), int> = 0>
#endif
    static auto make(M&& mean, Cov&& covariance) noexcept
    {
      return make_GaussianDistribution<C, random_number_engine>(std::forward<M>(mean), std::forward<Cov>(covariance));
    }

  };


  // ------------------------ //
  //        Overloads         //
  // ------------------------ //

  #ifdef __cpp_concepts
  template<gaussian_distribution Arg>
#else
  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  mean_of(Arg&& arg) noexcept
  {
    return (std::forward<Arg>(arg).mu);
  }


#ifdef __cpp_concepts
  template<gaussian_distribution Arg>
#else
  template<typename Arg, std::enable_if_t<gaussian_distribution<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  covariance_of(Arg&& arg) noexcept
  {
    return (std::forward<Arg>(arg).sigma);
  }


#ifdef __cpp_concepts
  template<gaussian_distribution T, std::size_t dimension = index_dimension_of_v<T, 0>,
      typename Scalar = scalar_type_of_t<T>, std::convertible_to<std::size_t>...runtime_dimensions> requires
    (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0))
#else
  template<typename T, std::size_t dimension = index_dimension_of<T, 0>::value,
    typename Scalar = typename scalar_type_of<T>::type, typename...runtime_dimensions, std::enable_if_t<
      gaussian_distribution<T> and (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0)) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  make_zero_distribution_like(runtime_dimensions...e)
  {
    using Coeffs = typename DistributionTraits<T>::FixedDescriptor;
    using re = DistributionTraits<T>::random_number_engine;
    auto m = make_zero<DistributionTraits<T>::Mean, Scalar>(Dimensions<dimension>{e...}, Dimension<1>{});
    auto c = make_zero<DistributionTraits<T>::Covariance, Scalar>(Dimensions<dimension>{e...}, Dimensions<dimension>{e...});
    return make_gaussian_distribution<re>>(std::move(m), std::move(c));
  }


#ifdef __cpp_concepts
  template<gaussian_distribution T, std::size_t dimension = index_dimension_of_v<T, 0>,
      typename Scalar = scalar_type_of_t<T>, std::convertible_to<std::size_t>...runtime_dimensions> requires
    (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0))
#else
  template<typename T, std::size_t dimension = index_dimension_of<T, 0>::value,
    typename Scalar = typename scalar_type_of<T>::type, typename...runtime_dimensions, std::enable_if_t<
      gaussian_distribution<T> and (sizeof...(runtime_dimensions) == (dimension == dynamic_size ? 1 : 0)) and
      (std::is_convertible_v<runtime_dimensions, std::size_t> and ...), int> = 0>
#endif
  constexpr auto
  make_normal_distribution_like(runtime_dimensions...e)
  {
    using Coeffs = typename DistributionTraits<T>::FixedDescriptor;
    using re = DistributionTraits<T>::random_number_engine;
    auto m = make_zero<DistributionTraits<T>::Mean, Scalar>(Dimensions<dimension>{e...}, Dimension<1>{});
    auto c = make_identity_matrix_like<DistributionTraits<T>::Covariance, Scalar>(Dimensions<dimension>{e...});
    return make_gaussian_distribution<re>>(std::move(m), std::move(c));
  }


#ifdef __cpp_concepts
  template<gaussian_distribution D, gaussian_distribution ... Ds>
#else
  template<typename D, typename ... Ds,
    std::enable_if_t<(gaussian_distribution<D> and ... and gaussian_distribution<Ds>), int> = 0>
#endif
  auto
  concatenate(D&& d, Ds&& ... ds)
  {
    if constexpr(sizeof...(Ds) > 0)
    {
      using re = typename DistributionTraits<D>::random_number_engine;
      auto m = concatenate(mean_of(std::forward<D>(d)), mean_of(std::forward<Ds>(ds))...);
      auto cov = concatenate(covariance_of(std::forward<D>(d)), covariance_of(std::forward<Ds>(ds))...);
      return make_GaussianDistribution<re>(std::move(m), std::move(cov));
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
      using re = typename DistributionTraits<Dist>::random_number_engine;
      return std::tuple {make_GaussianDistribution<re>(
        std::get<ints>(std::forward<Means>(ms)),
        std::get<ints>(std::forward<Covariances>(cs)))...};
    };
  }


  /// Split distribution.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor ... Cs, gaussian_distribution D> requires
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, typename DistributionTraits<D>::FixedDescriptor>
#else
  template<typename ... Cs, typename D, std::enable_if_t<
    (fixed_vector_space_descriptor<Cs> and ...) and gaussian_distribution<D> and
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, typename DistributionTraits<D>::FixedDescriptor>, int> = 0>
#endif
  inline auto
  split(D&& d) noexcept
  {
    using Coeffs = typename DistributionTraits<D>::FixedDescriptor;
    if constexpr(sizeof...(Cs) == 1 and equivalent_to<concatenate_fixed_vector_space_descriptor_t<Cs...>, Coeffs>)
    {
      return std::tuple(std::forward<D>(d));
    }
    else
    {
      auto means = split_vertical<Cs...>(mean_of(d));
      auto covariances = split_diagonal<Cs...>(covariance_of(std::forward<D>(d)));
      return OpenKalman::detail::zip_dist<D>(means, covariances, std::index_sequence_for<Cs...> {});
    }
  }


#ifdef __cpp_concepts
  template<gaussian_distribution Dist>
#else
  template<typename Dist, std::enable_if_t<gaussian_distribution<Dist>, int> = 0>
#endif
  inline std::ostream&
  operator<<(std::ostream& os, const Dist& d)
  {
    os << "mean:" << std::endl << mean_of(d) << std::endl <<
    "covariance:" << std::endl << covariance_of(d) << std::endl;
    return os;
  }


  // ---------------------- //
  //  Arithmetic Operators  //
  // ---------------------- //

#ifdef __cpp_concepts
  template<gaussian_distribution Dist1, gaussian_distribution Dist2> requires
    equivalent_to<typename DistributionTraits<Dist1>::FixedDescriptor, typename DistributionTraits<Dist2>::FixedDescriptor>
#else
  template<typename Dist1, typename Dist2, std::enable_if_t<
    gaussian_distribution<Dist1> and gaussian_distribution<Dist2> and
    equivalent_to<typename DistributionTraits<Dist1>::FixedDescriptor,
      typename DistributionTraits<Dist2>::FixedDescriptor>, int> = 0>
#endif
  inline auto
  operator+(Dist1&& d1, Dist2&& d2)
  {
    using re = typename DistributionTraits<Dist1>::random_number_engine;
    auto m1 = mean_of(std::forward<Dist1>(d1)) + mean_of(std::forward<Dist2>(d2));
    auto m2 = covariance_of(std::forward<Dist1>(d1)) + covariance_of(std::forward<Dist2>(d2));
    return make_GaussianDistribution<re>(std::move(m1), std::move(m2));
  };


#ifdef __cpp_concepts
  template<gaussian_distribution Dist1, gaussian_distribution Dist2> requires
    equivalent_to<typename DistributionTraits<Dist1>::FixedDescriptor, typename DistributionTraits<Dist2>::FixedDescriptor>
#else
  template<typename Dist1, typename Dist2, std::enable_if_t<
    gaussian_distribution<Dist1> and gaussian_distribution<Dist2> and
    equivalent_to<typename DistributionTraits<Dist1>::FixedDescriptor,
      typename DistributionTraits<Dist2>::FixedDescriptor>, int> = 0>
#endif
  inline auto
  operator-(Dist1&& d1, Dist2&& d2)
  {
    using re = typename DistributionTraits<Dist1>::random_number_engine;
    auto m1 = mean_of(std::forward<Dist1>(d1)) - mean_of(std::forward<Dist2>(d2));
    auto m2 = covariance_of(std::forward<Dist1>(d1)) - covariance_of(std::forward<Dist2>(d2));
    return make_GaussianDistribution<re>(std::move(m1), std::move(m2));
  };


#ifdef __cpp_concepts
  template<typed_matrix A, gaussian_distribution D> requires gaussian_distribution<D> and
    (not euclidean_transformed<A>) and
    (equivalent_to<vector_space_descriptor_of_t<A, 1>, typename DistributionTraits<D>::FixedDescriptor>)
#else
  template<typename A, typename D, std::enable_if_t<typed_matrix<A> and gaussian_distribution<D> and
    (not euclidean_transformed<A>) and
    (equivalent_to<vector_space_descriptor_of_t<A, 1>, typename DistributionTraits<D>::FixedDescriptor>), int> = 0>
#endif
  inline auto
  operator*(A&& a, D&& d)
  {
    using re = typename DistributionTraits<D>::random_number_engine;
    auto m = a * mean_of(std::forward<D>(d));
    auto c = scale(covariance_of(std::forward<D>(d)), std::forward<A>(a));
    return make_GaussianDistribution<re>(std::move(m), std::move(c));
  }


#ifdef __cpp_concepts
  template<gaussian_distribution Dist, std::convertible_to<const typename DistributionTraits<Dist>::Scalar> S>
#else
  template<typename Dist, typename S, std::enable_if_t<
    gaussian_distribution<Dist> and std::is_convertible_v<S, const typename DistributionTraits<Dist>::Scalar>, int> = 0>
#endif
  inline auto
  operator*(Dist&& d, S s)
  {
    using re = typename DistributionTraits<Dist>::random_number_engine;
    auto m = mean_of(std::forward<Dist>(d)) * s;
    auto c = scale(covariance_of(std::forward<Dist>(d)), s);
    return make_GaussianDistribution<re>(std::move(m), std::move(c));
  };


#ifdef __cpp_concepts
  template<gaussian_distribution Dist, std::convertible_to<const typename DistributionTraits<Dist>::Scalar> S>
#else
  template<typename Dist, typename S, std::enable_if_t<
    gaussian_distribution<Dist> and std::is_convertible_v<S, const typename DistributionTraits<Dist>::Scalar>, int> = 0>
#endif
  inline auto
  operator*(S s, Dist&& d)
  {
    using re = typename DistributionTraits<Dist>::random_number_engine;
    auto m = s * mean_of(std::forward<Dist>(d));
    auto c = scale(covariance_of(std::forward<Dist>(d)), s);
    return make_GaussianDistribution<re>(std::move(m), std::move(c));
  };


#ifdef __cpp_concepts
  template<gaussian_distribution Dist, std::convertible_to<const typename DistributionTraits<Dist>::Scalar> S>
#else
  template<typename Dist, typename S, std::enable_if_t<
    gaussian_distribution<Dist> and std::is_convertible_v<S, const typename DistributionTraits<Dist>::Scalar>, int> = 0>
#endif
  inline auto
  operator/(Dist&& d, S s)
  {
    using re = typename DistributionTraits<Dist>::random_number_engine;
    auto m = mean_of(std::forward<Dist>(d)) / s;
    auto c = inverse_scale(covariance_of(std::forward<Dist>(d)), s);
    return make_GaussianDistribution<re>(std::move(m), std::move(c));
  };

}

#endif //OPENKALMAN_GAUSSIANDISTRIBUTION_HPP
