/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_MONTECARLOTRANSFORM_H
#define OPENKALMAN_MONTECARLOTRANSFORM_H

#include <cmath>
#include "transforms/internal/TransformBase.h"
#include "distributions/GaussianDistribution.h"

namespace OpenKalman
{
  /**
   * @brief A linear transformation from one statistical distribution to another.
   * @tparam Dist The distribution type (e.g., Distribution).
   * @tparam Scalar The scalar type.
   * @tparam InputCoeffs Number of dimensions of the input distribution.
   * @tparam OutputCoeffs Number of dimensions of the output distribution.
   * @tparam noise_type Type of noise.
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs,
      NoiseType noise_type = NoiseType::additive,
      typename ... Noise>
  struct MonteCarloTransform;


  namespace
  {
    template<
        template<typename, typename> typename Dist,
        typename Scalar,
        typename InputCoeffs,
        typename OutputCoeffs>
    static inline std::tuple<const Dist<Scalar, OutputCoeffs>, const Eigen::Matrix<Scalar, InputCoeffs::size, OutputCoeffs::size>>
    MonteCarlo_transform(
        const std::function<Mean<Scalar, OutputCoeffs>(const Mean<Scalar, InputCoeffs>&)> g,
        const Dist<Scalar, InputCoeffs>& in_dist,
        const std::size_t samples)
    {
      using OutputMean = Mean<Scalar, OutputCoeffs>;
      using OutputMeanE = Eigen::Matrix<Scalar, OutputCoeffs::size_Euclidean, 1>;
      using CrossCovariance = Eigen::Matrix<Scalar, InputCoeffs::size, OutputCoeffs::size>;
      //
      // See B.P. Welford, Note on a Method for Calculating Corrected Sums of Squares and Products,
      // 4(3) Technometrics 419-20 (1962).
      const auto x_gen = SquareRootGaussianDistribution<Scalar, InputCoeffs> {in_dist};
      const auto Mx = mean(in_dist);
      OutputMean My = OutputMean::Zero();
      OutputMeanE MyE = OutputMeanE::Zero();
      Covariance<Scalar, OutputCoeffs> Myy = Eigen::Matrix<Scalar, OutputCoeffs::size, OutputCoeffs::size>::Zero();
      CrossCovariance Mxy = CrossCovariance::Zero();
      for (std::size_t n = 1; n <= samples; n++)
      {
        const auto x = x_gen();
        const auto y = g(x);
        const auto yE = Mean<Scalar, OutputCoeffs>::to_Euclidean(y);
        const auto delta1 = y - My; // Wraps automatically.
        MyE += (yE - MyE) / n;
        My = Mean<Scalar, OutputCoeffs>::from_Euclidean(MyE);
        const auto delta2 = y - My; // Wraps automatically.
        Myy += delta1 * delta2.adjoint();
        Mxy += (x - Mx) * delta2.adjoint(); // Wraps automatically.
      };
      const Scalar n = samples - 1;
      return {{My, Myy / n}, Mxy / n};
    }
  }

  /**
   * @brief Noiseless transform.
   * @tparam Dist The distribution type (e.g., Distribution).
   * @tparam Scalar The scalar type.
   * @tparam InputCoeffs Number of dimensions of the input distribution.
   * @tparam OutputCoeffs Number of dimensions of the output distribution.
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs>
  struct MonteCarloTransform<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::none>
      : TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::none>
  {
    using InputDist = Dist<Scalar, InputCoeffs>; // Input distribution type
    using OutputDist = Dist<Scalar, OutputCoeffs>; // Output distribution type
    using CrossCovariance = Eigen::Matrix<Scalar, InputCoeffs::size, OutputCoeffs::size>; // Cross-covariance type
    using Output = std::tuple<const OutputDist, const CrossCovariance>; // Output type
    using Base = TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::none>;

    /**
     * @brief Construct transform from input distribution to output
     * @param g A transformation.
     */
    template<
        template<typename, typename, typename, NoiseType, typename...> typename Trans,
        typename ... Args>
    explicit MonteCarloTransform(
        const Trans<Scalar, InputCoeffs, OutputCoeffs, NoiseType::none, Args ...>& g,
        const std::size_t samples = 100000)
        : Base
              {
                  [g, samples](const InputDist& x) -> Output
                  {
                    return MonteCarlo_transform(g, x, samples);
                  }
              } {}

  };


  /**
   * @brief Transform with additive noise.
   * @tparam Dist The distribution type (e.g., Distribution).
   * @tparam Scalar The scalar type.
   * @tparam InputCoeffs Number of dimensions of the input distribution.
   * @tparam OutputCoeffs Number of dimensions of the output distribution.
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs,
      typename ... Noise>
  struct MonteCarloTransform<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::additive, Noise...>
      : TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::additive, Noise...>
  {
    using InputDist = Dist<Scalar, InputCoeffs>; // Input distribution type
    using OutputDist = Dist<Scalar, OutputCoeffs>; // Output distribution type
    using CrossCovariance = Eigen::Matrix<Scalar, InputCoeffs::size, OutputCoeffs::size>; // Cross-covariance type
    using Output = std::tuple<const OutputDist, const CrossCovariance>; // Output type
    using Base = TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::additive, Noise...>;

    /**
     * @brief Constructs transform that adds noise after the transform.
     * @param f Function from InputDist to Output
     * @param add_noise Noise-adding function.
     */
    template<
        template<typename, typename, typename, NoiseType, typename...> typename Transformation,
        typename ... Args>
    explicit MonteCarloTransform(
        const Transformation<Scalar, InputCoeffs, OutputCoeffs, NoiseType::additive, Args ...>& g,
        const std::size_t samples = 100000)
        : Base
              {
                  [g, samples](const InputDist& x, const Noise& ... n) -> Output
                  {
                    const std::function<Mean<Scalar, OutputCoeffs>(const Mean<Scalar, InputCoeffs>&)> g2 =
                        [&g, n...](const Mean<Scalar, InputCoeffs>& x) { return g(x, mean(n)...); };
                    return MonteCarlo_transform(g2, x, samples);
                  }
              } {}

  };


  /**
   * @brief Transform where input is augmented with noise.
   * @tparam Dist The distribution type (e.g., Distribution).
   * @tparam Scalar The scalar type.
   * @tparam InputCoeffs Number of dimensions of the input distribution.
   * @tparam OutputCoeffs Number of dimensions of the output distribution.
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs,
      typename ... Noise>
  struct MonteCarloTransform<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::augmented, Noise...>
      : TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::augmented, Noise...>
  {
    using InputDist = Dist<Scalar, InputCoeffs>; // Input distribution type
    using OutputDist = Dist<Scalar, OutputCoeffs>; // Output distribution type
    using AugmentedDist = Dist<Scalar, Concatenate<InputCoeffs, typename Noise::Coefficients...>>; // Augmented input distribution type
    using CrossCovariance = Eigen::Matrix<Scalar, InputCoeffs::size, OutputCoeffs::size>; // Cross-covariance type
    using Output = std::tuple<const OutputDist, const CrossCovariance>; // Output type
    using Base = TransformBase<Dist, Scalar, InputCoeffs, OutputCoeffs, NoiseType::augmented, Noise...>;

    /**
     * @brief Constructor that augments the input with the noise.
     * @param f Function from AugmentedInput to Output
     * @param augment Augmentation function.
     */
    template<
        template<typename, typename, typename, NoiseType, typename...> typename Transformation,
        typename ... Args>
    explicit MonteCarloTransform(
        const Transformation<Scalar, InputCoeffs, OutputCoeffs, NoiseType::augmented, Args ...>& g,
        const std::size_t samples = 100000)
        : Base
              {
                  [g, samples](const AugmentedDist& x_aug) -> Output
                  {
                    const auto xn = split<Dist, Scalar, InputCoeffs, typename Noise::Coefficients...>(x_aug);
                    const auto x = std::get<0>(xn);
                    const std::function<Mean<Scalar, OutputCoeffs>(const Mean<Scalar, InputCoeffs>&)> g2 =
                        [&](const Mean<Scalar, InputCoeffs>& x_r)
                        {
                          return std::apply(
                              g,
                              std::tuple_cat(
                                  std::tuple{x_r},
                                  std::apply([](const auto&, const auto& ... n) {return std::tuple {mean(n)...};}, xn)));
                        };
                    return MonteCarlo_transform(g2, x, samples);
                  }
              } {}

  };


  /**
   * Deduction guide
   */
  /// Infer GaussianDistribution
  template<
      template<typename, typename, typename, NoiseType, typename...> typename Transformation,
      typename ... Args,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs>
  MonteCarloTransform(
      const Transformation<Scalar, InputCoeffs, OutputCoeffs, NoiseType::none, Args ...>&,
      const int)
  -> MonteCarloTransform<GaussianDistribution, Scalar, InputCoeffs, OutputCoeffs, NoiseType::none>;

  template<
      template<typename, typename, typename, NoiseType, typename...> typename Transformation,
      typename ... Args,
      typename Scalar,
      typename InputCoeffs,
      typename OutputCoeffs,
      NoiseType noise_type,
      typename = std::enable_if_t<noise_type != NoiseType::none>>
  MonteCarloTransform(
      const Transformation<Scalar, InputCoeffs, OutputCoeffs, noise_type, Args ...>&,
      const int)
  -> MonteCarloTransform<GaussianDistribution, Scalar, InputCoeffs, OutputCoeffs, noise_type, GaussianDistribution<Scalar, OutputCoeffs>>;

}


#endif //OPENKALMAN_MONTECARLOTRANSFORM_H
