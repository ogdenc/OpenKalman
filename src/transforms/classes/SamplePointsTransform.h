/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
#define OPENKALMAN_SAMPLEPOINTSTRANSFORM_H

#include <functional>
#include <tuple>

namespace OpenKalman
{
  /**
   * @brief Scaled points transform. Compatible with unscented transform and cubature transform.
   * As implemented in, e.g.,
   * E. Wan & R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
   * in Proc. of IEEE Symposium (AS-SPCC), pp. 153-158.
   * See also R. van der Merwe & E. Wan, "The Square-Root Unscented Kalman Filter for State and
   * Parameter-Estimation in Proc. Acoustics, Speech, and Signal Processing (ICASSP'01), 2001, pp. 3461-64.
   */
  template<
    /// The type of sample points on which the transform is based.
    typename SamplePointsType>
  struct SamplePointsTransform;


  using CubatureTransform = SamplePointsTransform<CubaturePoints>;

  using UnscentedTransform = SamplePointsTransform<UnscentedSigmaPoints>;


  template<typename SamplePointsType>
  struct SamplePointsTransform
  {
  private:
    template<typename Transformation, typename...XDevs, typename...Dists, std::size_t...ints>
    constexpr auto y_means_impl(
      const Transformation& g,
      const std::tuple<XDevs...>& x_devs,
      const std::tuple<Dists...>& dists,
      std::index_sequence<ints...>) const
    {
      constexpr auto count = MatrixTraits<decltype(std::get<0>(x_devs))>::columns;
      return apply_columnwise<count>([&](size_t i) {
        return to_Euclidean(g((column(std::get<ints>(x_devs), i) + make_Matrix(mean(std::get<ints>(dists))))...));
      });
    }

  public:
    /**
     * Perform a sample points transform from one statistical distribution to another.
     * @tparam Transformation The transformation on which the transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<typename Transformation, typename InputDist, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto operator()(const Transformation& g, const InputDist& x, const NoiseDist& ...n) const
    {
      // The sample points, divided into tuples for the input and each noise term:
      const auto sample_points_tuple = SamplePointsType::sample_points(x, n...);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      //
      auto xpoints = std::get<0>(sample_points_tuple);
      // Calculate y means for each sigma point:
      auto ymeans = y_means_impl(
        g, sample_points_tuple, std::forward_as_tuple(x, n...), std::make_index_sequence<sizeof...(NoiseDist) + 1>());
      //
      auto mean_output = strict(SamplePointsType::template weighted_means<dim>(ymeans));
      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints = apply_columnwise(ymeans, [&mean_output](const auto& col) { return col - mean_output; });

      auto [out_covariance, cross_covariance] = SamplePointsType::template covariance<dim, InputDist>(xpoints, ypoints);
      auto out = GaussianDistribution {mean_output, out_covariance};
      return std::tuple {std::move(out), std::move(cross_covariance)};
    }

    /**
     * Perform two sample points transforms.
     * @tparam Transformation1 The transformation on which the first transform is based.
     * @tparam Transformation2 The transformation on which the second transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<typename Transformation1, typename Transformation2, typename InputDist, typename QNoise, typename RNoise,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<QNoise>, is_distribution<RNoise>>, int> = 0>
    auto operator()(const Transformation1& g1, const Transformation2& g2, const InputDist& x, const QNoise& q, const RNoise& r) const
    {
      // The scaled sample points, divided into tuples for the input and each noise term:
      const auto [xpoints1, xpoints2, xpoints3] = SamplePointsType::sample_points(x, q, r);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + DistributionTraits<QNoise>::dimension + DistributionTraits<RNoise>::dimension);

      // First transform:
      auto ymeans = y_means_impl(g1, std::forward_as_tuple(xpoints1, xpoints2), std::forward_as_tuple(x, q), std::make_index_sequence<2>());
      auto mean_output1 = SamplePointsType::template weighted_means<dim>(ymeans);
      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints1 = apply_columnwise(ymeans, [&mean_output1](const auto& col) { return col - mean_output1; });
      auto [out_covariance1, _] = SamplePointsType::template covariance<dim, InputDist>(xpoints1, ypoints1);
      auto out1 = GaussianDistribution {mean_output1, out_covariance1};

      // Second transform:
      auto zmeans = y_means_impl(g2, std::forward_as_tuple(ypoints1, xpoints3), std::forward_as_tuple(out1, r), std::make_index_sequence<2>());
      auto mean_output2 = strict(SamplePointsType::template weighted_means<dim>(zmeans));
      // Each column is a deviation from y mean for each transformed sigma point:
      auto zpoints1 = apply_columnwise(zmeans, [&mean_output2](const auto& col) { return col - mean_output2; });
      auto [out_covariance2, cross_covariance] = SamplePointsType::template covariance<dim, InputDist>(xpoints1, zpoints1);
      auto out2 = GaussianDistribution {mean_output2, out_covariance2};

      return std::tuple {std::move(out2), std::move(cross_covariance)};
    }

  };


}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
