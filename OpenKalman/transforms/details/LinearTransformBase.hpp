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
 * \internal
 * \brief Definition of internal::LinearTransformBase and internal::needs_additive_correction
 */

#ifndef OPENKALMAN_LINEARTRANSFORMBASE_HPP
#define OPENKALMAN_LINEARTRANSFORMBASE_HPP

#include <functional>
#include <tuple>
#include "collections/concepts/uniformly_gettable.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Whether tests T needs an additive correction when F is the tests function.
   * \details This is true for LinearizedTransform<order> where order >= 2.
   */
  template<typename T, typename F>
  struct needs_additive_correction : std::false_type {};


  /**
   * \internal
   * /brief Base class for linear or linearized transformations from one statistical distribution to another.
   * /details Each class deriving from LinearTransformBase must define TransformModel within class scope.
   * TransformModel is the underlying transform function for the derived class, which takes an input distribution
   * and an optional set of noise distributions and returns a tuple comprising the following, used in constructing the
   * output distribution and cross-covariance:
   * -# the output mean, and
   * -# a tuple of Jacobians corresponding to each input and noise term.
   **/
  template<typename Derived>
  struct LinearTransformBase : TransformBase<Derived>
  {

  private:

    using Base = TransformBase<Derived>;

    template<std::size_t return_cross, typename J, typename D, std::size_t...ints>
    static auto sum_noise_terms(const J& j, D&& d, std::index_sequence<ints...>)
    {
      using InputDist = collections::collection_element_t<0, D>;
      if constexpr(cholesky_form<InputDist>)
      {
        if constexpr (return_cross)
        {
          auto sqrt_c0 =square_root(covariance_of(std::get<0>(std::forward<D>(d))));
          auto term0 = std::get<0>(j) * sqrt_c0;
          return std::tuple {
            make_self_contained(LQ_decomposition(concatenate_horizontal(term0, Matrix(
              (std::get<ints+1>(j) * (square_root(covariance_of(std::get<ints+1>(std::forward<D>(d)))))))...))),
            make_self_contained(std::move(sqrt_c0) * conjugate_transpose(term0))};
        }
        else
        {
          return make_self_contained(LQ_decomposition(concatenate_horizontal(
            std::get<0>(j) * square_root(covariance_of(std::get<0>(std::forward<D>(d)))),
            Matrix((std::get<ints+1>(j) * (square_root(covariance_of(std::get<ints+1>(std::forward<D>(d)))))))...)));
        }
      }
      else
      {
        if constexpr (return_cross)
        {
          auto cross = make_self_contained(covariance_of(std::get<0>(std::forward<D>(d))) * conjugate_transpose(std::get<0>(j)));
          auto cov = make_covariance(make_self_contained((
            (std::get<0>(j) * cross) +
              ... + (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(std::forward<D>(d))) * conjugate_transpose(std::get<ints+1>(j)))))));
          return std::tuple {std::move(cov), std::move(cross)};
        }
        else
        {
          return make_covariance(make_self_contained((
            (std::get<0>(j) * covariance_of(std::get<0>(std::forward<D>(d))) * conjugate_transpose(std::get<0>(j))) + ... +
            (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(std::forward<D>(d))) * conjugate_transpose(std::get<ints+1>(j)))))));
        }
      }
    }


    /*
     * Linearly transform one statistical distribution to another.
     * \tparam Trans The linear or linearized tests on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam InputDist The prior distribution.
     * \tparam NoiseDist Zero or more noise distributions.
     **/
#ifdef __cpp_concepts
    template<std::size_t return_cross, linearized_function<1> Trans,
      gaussian_distribution InputDist, gaussian_distribution ... NoiseDists>
#else
    template<std::size_t return_cross, typename  Trans, typename InputDist, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      linearized_function<Trans, 1>, int> = 0>
#endif
    auto transform(const Trans& g, const InputDist& in, const NoiseDists& ...n) const
    {
      typename Derived::template TransformModel<Trans> transform_model {g};
      auto [mean_output, jacobians] = transform_model(mean_of(in), mean_of(n)...);
      using re = typename DistributionTraits<InputDist>::random_number_engine;

      if constexpr (return_cross)
      {
        auto [cov_out, cross_covariance] = sum_noise_terms<true>(jacobians, std::forward_as_tuple(in, n...),
          std::make_index_sequence<std::min(sizeof...(NoiseDists), collections::size_of_v<decltype(jacobians)> - 1)> {});

        auto out = make_GaussianDistribution<re>(std::move(mean_output), std::move(cov_out));

        if constexpr(needs_additive_correction<Derived, Trans>::value)
          return std::tuple {make_self_contained(out + transform_model.add_correction(in, n...)), cross_covariance};
        else
          return std::tuple {out, cross_covariance};
      }
      else
      {
        auto cov_out = sum_noise_terms<false>(jacobians, std::forward_as_tuple(in, n...),
          std::make_index_sequence<std::min(sizeof...(NoiseDists), collections::size_of_v<decltype(jacobians)> - 1)> {});

        auto out = make_GaussianDistribution<re>(std::move(mean_output), std::move(cov_out));

        if constexpr(needs_additive_correction<Derived, Trans>::value)
        {
          return make_self_contained(out + transform_model.add_correction(in, n...));
        }
        else
        {
          return out;
        }
      }
    }

  public:

    /**
     * \brief Perform one or more consecutive linear(ized) transforms.
     * \tparam InputDist The prior distribution.
     * \tparam Ts A list of tuple-like structures, each containing arguments to a transform.
     * These arguments each include a tests and zero or more noise distributions.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, collections::uniformly_gettable...Ts>
#else
    template<typename InputDist, typename...Ts, std::enable_if_t<
      gaussian_distribution<InputDist> and (collections::uniformly_gettable<Ts> and ...), int> = 0>
#endif
    auto operator()(const InputDist& x, const Ts&...ts) const
    {
      return Base::operator()(x, ts...);
    }


    /**
     * \brief Perform a linear(ized) transform from one statistical distribution to another.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The linear or linearized tests on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam NoiseDists Zero or more noise distributions.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, linearized_function<1> Trans, gaussian_distribution ... NoiseDists>
      requires requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      linearized_function<Trans, 1> and std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto operator()(const InputDist& x, const Trans& g, const NoiseDists&...ns) const
    {
      return transform<false>(g, x, ns...);
    }


    /**
     * \brief Perform one or more consecutive linear(ized) transforms, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Ts A list of tuple-like structures, each containing arguments to a transform.
     * These arguments each include a tests and zero or more noise distributions.
     * \return A tuple containing the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, collections::uniformly_gettable...Ts>
#else
    template<typename InputDist, typename...Ts, std::enable_if_t<
      gaussian_distribution<InputDist> and (collections::uniformly_gettable<Ts> and ...), int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const Ts&...ts) const
    {
      return Base::transform_with_cross_covariance(x, ts...);
    }


    /**
     * \brief Perform a linear(ized) transform, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The linear or linearized tests on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam NoiseDists Zero or more noise distributions.
     * \return A tuple comprising the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, linearized_function<1> Trans, gaussian_distribution ... NoiseDists>
      requires requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      linearized_function<Trans, 1> and std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto transform_with_cross_covariance(const InputDist& x, const Trans& g, const NoiseDists&...ns) const
    {
      return transform<true>(g, x, ns...);
    }

  };


}

#endif
