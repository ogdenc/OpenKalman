/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
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
   * Internal base class for linear or linearized transformations from one statistical distribution to another.
   **/
  template<typename Derived>
  struct LinearTransformBase : TransformBase<Derived>
  {
  private:
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
          auto cov = make_Covariance((
            (std::get<0>(j) * cross) +
              ... + (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(dist)) * adjoint(std::get<ints+1>(j))))));
          return std::tuple {std::move(cov), std::move(cross)};
        }
        else
        {
          return make_Covariance((
            (std::get<0>(j) * covariance_of(std::get<0>(dist)) * adjoint(std::get<0>(j))) +
              ... + (std::get<ints+1>(j) * (covariance_of(std::get<ints+1>(dist)) * adjoint(std::get<ints+1>(j))))));
        }
      }
    }

  protected:
  using Base = TransformBase<Derived>;

  /**
   * Linearly transform one statistical distribution to another.
   * \tparam TransformFunction Underlying transform function that takes an input distribution and an optional set of
   * noise distributions and returns the following information used in constructing the output distribution
   * and cross-covariance:
   * -# the output mean, and
   * -# a tuple of Jacobians corresponding to each input and noise term,
   * \tparam InputDist Input distribution.
   * \tparam NoiseDist Noise distribution.
   **/
    template<std::size_t return_cross, typename TransformFunction, typename InputDist, typename ... NoiseDists,
      std::enable_if_t<(distribution<InputDist> and ... and distribution<NoiseDists>), int> = 0>
    auto transform(const TransformFunction& f, const InputDist& in, const NoiseDists& ...n) const
    {
      auto[mean_output, jacobians] = f(mean_of(in), mean_of(n)...);
      if constexpr (return_cross)
      {
        auto [cov_out, cross_covariance] = sum_noise_terms<true>(jacobians, std::tuple {in, n...},
          std::make_index_sequence<std::min(sizeof...(NoiseDists), std::tuple_size_v<decltype(jacobians)> - 1)>{});
        auto out = make_GaussianDistribution(mean_output, cov_out);
        if constexpr(TransformFunction::correction)
          return std::tuple {make_self_contained(out + f.add_correction(in, n...)), cross_covariance};
        else
          return std::tuple {out, cross_covariance};
      }
      else
      {
        auto cov_out = sum_noise_terms<false>(jacobians, std::tuple {in, n...},
          std::make_index_sequence<std::min(sizeof...(NoiseDists), std::tuple_size_v<decltype(jacobians)> - 1)>{});
        auto out = make_GaussianDistribution(mean_output, cov_out);
        if constexpr(TransformFunction::correction)
          return make_self_contained(out + f.add_correction(in, n...));
        else
          return out;
      }
    }

  public:
    using Base::operator();
    using Base::transform_with_cross_covariance;

    /**
     * Perform a linear(ized) transform from one statistical distribution to another.
     * \tparam InputDist Input distribution.
     * \tparam Transformation The transformation on which the transform is based (e.g., LinearTransformation).
     * \tparam NoiseDists Noise distributions.
     **/
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (distribution<InputDist> and ... and distribution<NoiseDists>) and is_linearized_function_v<Trans, 1>, int> = 0>
    auto operator()(const InputDist& x, const Trans& g, const NoiseDists&...ns) const
    {
      return transform<false>(typename Derived::template TransformFunction<Trans> {g}, x, ns...);
    }

    /**
     * Perform a linear(ized) transform, also returning the cross-covariance.
     * \tparam InputDist Input distribution.
     * \tparam Transformation The transformation on which the transform is based (e.g., LinearTransformation).
     * \tparam NoiseDists Noise distributions.
     **/
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (distribution<InputDist> and ... and distribution<NoiseDists>) and is_linearized_function_v<Trans, 1>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const Trans& g, const NoiseDists&...ns) const
    {
      return transform<true>(typename Derived::template TransformFunction<Trans> {g}, x, ns...);
    }

  };


}

#endif //OPENKALMAN_LINEARTRANSFORMBASE_HPP
