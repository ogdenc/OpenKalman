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
    template<typename J, typename Dist, std::size_t...ints>
    static auto sum_noise_terms(const J& j, const Dist& dist, std::index_sequence<ints...>)
    {
      using InputDist = std::tuple_element<0, Dist>;
      if constexpr(is_Cholesky_v<InputDist>)
      {
        const auto sqrt_c0 =square_root(covariance(std::get<0>(dist)));
        const auto term0 = std::get<0>(j) * sqrt_c0;
        return std::tuple {
          strict(sqrt_c0 * adjoint(term0)),
          LQ_decomposition(concatenate_horizontal(term0, TypedMatrix(
            (std::get<ints+1>(j) * (square_root(covariance(std::get<ints+1>(dist))))))...))};
      }
      else
      {
        const auto cross = strict(covariance(std::get<0>(dist)) * adjoint(std::get<0>(j)));
        return std::tuple {
          cross,
          make_Covariance(strict((
          (std::get<0>(j) * cross) +
            ... + (std::get<ints+1>(j) * (covariance(std::get<ints+1>(dist)) * adjoint(std::get<ints+1>(j)))))))};
      }
    }

  public:
    template<typename InputDist, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      const auto[mean_output, jacobians] = function(mean(in), mean(n)...);
      const auto j0 = std::get<0>(jacobians);
      auto [cross_covariance, cov_out] = sum_noise_terms(jacobians, std::tuple {in, n...},
        std::make_index_sequence<std::min(sizeof...(NoiseDist), std::tuple_size_v<decltype(jacobians)> - 1)>{});
      const auto out = make_GaussianDistribution(mean_output, cov_out);
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
