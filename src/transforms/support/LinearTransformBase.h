/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_LINEARTRANSFORMBASE_H
#define OPENKALMAN_LINEARTRANSFORMBASE_H

#include <functional>
#include <tuple>
#include <optional>
#include <Eigen/Dense>

namespace OpenKalman::internal
{

  /**
   * Internal base class for linear or linearized transformations from one statistical distribution to another.
   * @tparam InputCoefficients Coefficient types for input distribution.
   * @tparam OutputCoefficients Coefficient types for the output distribution.
   * @tparam TransformFunction Underlying transform function that takes an input distribution and an optional set of
   * noise distributions and returns the following information used in constructing the output distribution
   * and cross-covariance:
   * -# the output mean, and
   * -# a tuple of Jacobians corresponding to each input and noise term,
   **/
  template<typename InputCoefficients, typename OutputCoefficients, typename TransformFunction>
  struct LinearTransformBase;


  template<
    typename InputCoeffs,
    typename OutputCoeffs,
    typename TransformFunction>
  struct LinearTransformBase
  {
    using InputCoefficients = InputCoeffs;
    using OutputCoefficients = OutputCoeffs;

    const TransformFunction function;

    LinearTransformBase(const TransformFunction& f) : function {f} {}

    LinearTransformBase(TransformFunction&& f) noexcept : function {std::move(f)} {}

  private:
    template<typename J, typename C, typename T0, std::size_t...ints>
    static auto sum_noise_terms(const J& j, const C& cov, T0 term0, std::index_sequence<ints...>)
    {
      return make_Covariance(strict((term0 + ... + (std::get<ints>(j) * (std::get<ints>(cov) * adjoint(std::get<ints>(j)))))));
    }

    template<typename J, typename SqC, typename T0, std::size_t...ints>
    static auto sum_sqrt_noise_terms(const J& j, const SqC& cov, T0 term0, std::index_sequence<ints...>)
    {
      return LQ_decomposition(concatenate_horizontal(term0, (std::get<ints>(j) * (std::get<ints>(cov)))...));
    }

  public:
    template<typename InputDist, typename ... NoiseDist,
      std::enable_if_t<not std::disjunction_v<
        is_Cholesky<typename DistributionTraits<InputDist>::Covariance>,
        is_Cholesky<typename DistributionTraits<NoiseDist>::Covariance>...>, int> = 0>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      const auto[mean_output, jacobians] = function(mean(in), mean(n)...);
      const auto j0 = std::get<0>(jacobians);
      const auto c0 = covariance(in);
      const auto cross_covariance = strict(c0 * adjoint(j0));
      const auto cov_out = sum_noise_terms(
        jacobians,
        std::tuple(0, covariance(n)...), // The 0 is a dummy value.
        j0 * cross_covariance,
        std::make_index_sequence<std::min(sizeof...(NoiseDist) + 1, std::tuple_size_v<decltype(jacobians)>)>{});
      const auto out = make_GaussianDistribution(mean_output, cov_out);
      if constexpr(TransformFunction::correction)
        return std::tuple {strict(out + function.add_correction(in, n...)), cross_covariance};
      else
        return std::tuple {std::move(out), cross_covariance};
    }

    template<typename InputDist, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<
        is_Cholesky<typename DistributionTraits<InputDist>::Covariance>,
        is_Cholesky<typename DistributionTraits<NoiseDist>::Covariance>...>, int> = 0>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      const auto[mean_output, jacobians] = function(mean(in), mean(n)...);
      const auto j0 = std::get<0>(jacobians);
      const auto sqrt_c0 = square_root(covariance(in));
      const auto j0_sqrt_c0 = j0 * sqrt_c0;
      const auto cov_out = sum_sqrt_noise_terms(
        jacobians,
        std::tuple(0, square_root(covariance(n))...), // The 0 is a dummy value.
        j0_sqrt_c0,
        std::make_index_sequence<std::min(sizeof...(NoiseDist) + 1, std::tuple_size_v<decltype(jacobians)>)>{});
      const auto out = make_GaussianDistribution(mean_output, cov_out);
      const auto cross_covariance = strict(sqrt_c0 * adjoint(j0_sqrt_c0));
      if constexpr(TransformFunction::correction)
        return std::tuple {strict(out + function.add_correction(in, n...)), cross_covariance};
      else
        return std::tuple {out, cross_covariance};
    }

  };


  /// Composition of two Transform objects.
  template<typename Transform1, typename Transform2>
  struct Composition
  {
    Composition(const Transform1& t1, const Transform2& t2) : transform1(t1), transform2(t2) {}

    Transform1 transform1;
    Transform2 transform2;

    template<
      typename InputDist,
      typename ... NonlinearNoise1, typename ... LinearNoise1,
      typename ... NonlinearNoise2, typename ... LinearNoise2>
    constexpr auto operator()(
      const InputDist& in,
      const std::tuple<NonlinearNoise1...>& n1,
      const std::tuple<LinearNoise1& ...>& l1,
      const std::tuple<NonlinearNoise2...>& n2,
      const std::tuple<LinearNoise2& ...>& l2)
    {
      auto[out1, cross1] = std::apply(transform1, std::tuple_cat(std::tuple(std::tuple_cat(std::tuple(in), n1)), l1));
      return std::apply(transform2, std::tuple_cat(std::tuple(std::tuple_cat(std::tuple(out1), n2)), l2));
    }

  };

}

#endif //OPENKALMAN_LINEARTRANSFORMBASE_H
