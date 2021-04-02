/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMBASE_HPP
#define OPENKALMAN_LINEARTRANSFORMBASE_HPP

#include <functional>
#include <tuple>


namespace OpenKalman::internal
{

  /**
   * \brief Whether transformation T needs an additive correction when F is the transformation function.
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
  struct LinearTransformBase : internal::TransformBase<Derived>
  {

  private:

    using Base = internal::TransformBase<Derived>;

    template<std::size_t return_cross, typename J, typename Dist, std::size_t...ints>
    static auto sum_noise_terms(const J& j, const Dist& dist, std::index_sequence<ints...>)
    {
      using InputDist = std::tuple_element<0, Dist>;
      if constexpr(cholesky_form<InputDist>)
      {
        if constexpr (return_cross)
        {
          auto sqrt_c0 =square_root(covariance_of(std::get<0>(dist)));
          auto term0 = std::get<0>(j) * sqrt_c0;
          return std::tuple {
            LQ_decomposition(concatenate_horizontal(term0, Matrix(
              (std::get<ints+1>(j) * (square_root(covariance_of(std::get<ints+1>(dist))))))...)),
            make_self_contained(sqrt_c0 * adjoint(term0))};
        }
        else
        {
          auto term0 = std::get<0>(j) * square_root(covariance_of(std::get<0>(dist)));
          return LQ_decomposition(concatenate_horizontal(term0, Matrix(
              (std::get<ints+1>(j) * (square_root(covariance_of(std::get<ints+1>(dist))))))...));
        }
      }
      else
      {
        if constexpr (return_cross)
        {
          auto cross = make_self_contained(covariance_of(std::get<0>(dist)) * adjoint(std::get<0>(j)));
          auto cov = make_covariance((
            (std::get<0>(j) * cross) +
              ... + (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(dist)) * adjoint(std::get<ints+1>(j))))));
          return std::tuple {std::move(cov), std::move(cross)};
        }
        else
        {
          return make_covariance((
            (std::get<0>(j) * covariance_of(std::get<0>(dist)) * adjoint(std::get<0>(j))) +
              ... + (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(dist)) * adjoint(std::get<ints+1>(j))))));
        }
      }
    }


    /*
     * Linearly transform one statistical distribution to another.
     * \tparam Trans The linear or linearized transformation on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam InputDist Input distribution.
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
      auto[mean_output, jacobians] = transform_model(mean_of(in), mean_of(n)...);

      if constexpr (return_cross)
      {
        auto [cov_out, cross_covariance] = sum_noise_terms<true>(jacobians, std::tuple {in, n...},
          std::make_index_sequence<std::min(sizeof...(NoiseDists), std::tuple_size_v<decltype(jacobians)> - 1)>{});
        auto out = make_GaussianDistribution(mean_output, cov_out);

        if constexpr(needs_additive_correction<Derived, Trans>::value)
          return std::tuple {make_self_contained(out + transform_model.add_correction(in, n...)), cross_covariance};
        else
          return std::tuple {out, cross_covariance};
      }
      else
      {
        auto cov_out = sum_noise_terms<false>(jacobians, std::tuple {in, n...},
          std::make_index_sequence<std::min(sizeof...(NoiseDists), std::tuple_size_v<decltype(jacobians)> - 1)>{});
        auto out = make_GaussianDistribution(mean_output, cov_out);

        if constexpr(needs_additive_correction<Derived, Trans>::value)
          return make_self_contained(out + transform_model.add_correction(in, n...));
        else
          return out;
      }
    }

  public:

    using Base::operator();


    /**
     * Perform a linear(ized) transform from one statistical distribution to another.
     * \tparam InputDist Input distribution.
     * \tparam Trans The linear or linearized transformation on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam NoiseDists Zero or more noise distributions.
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


    using Base::transform_with_cross_covariance;


    /**
     * Perform a linear(ized) transform, also returning the cross-covariance.
     * \tparam InputDist Input distribution.
     * \tparam Trans The linear or linearized transformation on which the transform is based
     * (e.g., LinearTransformation).
     * \tparam NoiseDists Zero or more noise distributions.
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

#endif //OPENKALMAN_LINEARTRANSFORMBASE_HPP
