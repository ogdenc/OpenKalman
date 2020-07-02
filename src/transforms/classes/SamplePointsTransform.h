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
#include <Eigen/Dense>

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
    typename SamplePointsType,
    /// The transformation on which the transform is based.
    typename TransformationType>
  struct SamplePointsTransform;


  template<typename TransformationType>
  using CubatureTransform = SamplePointsTransform<CubaturePoints, TransformationType>;

  template<typename TransformationType>
  using UnscentedTransform = SamplePointsTransform<UnscentedSigmaPoints, TransformationType>;


  template<typename SamplePointsType, typename TransformationType>
  struct SamplePointsTransform
  {
    using InputCoefficients = typename TransformationType::InputCoefficients;
    using OutputCoefficients = typename TransformationType::OutputCoefficients;

    explicit SamplePointsTransform(const TransformationType& transformation)
      : transformation(transformation) {}

  protected:
    const TransformationType transformation;

  private:
    template<typename...XDevs, typename...Dists, std::size_t...ints>
    constexpr auto y_means_impl(
      const std::tuple<XDevs...>& x_devs,
      const std::tuple<Dists...>& dists,
      std::index_sequence<ints...>) const
    {
      constexpr auto count = MatrixTraits<decltype(std::get<0>(x_devs))>::columns;
      return apply_columnwise<count>([&x_devs, &dists, this](size_t i) {
        return to_Euclidean(transformation((column(std::get<ints>(x_devs), i) + make_Matrix(mean(std::get<ints>(dists))))...));
      });
    }

  protected:
    /**
     * Underlying transform function that takes an input distribution and an optional set of
     * noise distributions and returns the following information used in constructing the output distribution
     * and cross-covariance:
     * -# the output mean,
     * -# the x-deviations, and
     * -# the y-deviations.
     */
    template<typename Dist, typename ... Noise>
    auto trans(const Dist& x, const Noise& ... n) const
    {
      // The sample points, divided into tuples for the input and each noise term:
      const auto sample_points_tuple = SamplePointsType::sample_points(x, n...);
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ... + DistributionTraits<Noise>::dimension);
      //
      auto x_deviations = std::get<0>(sample_points_tuple);
      auto y_means = y_means_impl(sample_points_tuple, std::tuple{x, n...}, std::make_index_sequence<sizeof...(Noise) + 1>());
      std::cout << "y_means" << std::endl << y_means << std::endl << std::flush;
      //
      auto mean_output = strict(SamplePointsType::template weighted_means<dim>(y_means));
      // Each column is a deviation from y mean for each transformed sigma point:
      auto y_deviations = apply_columnwise(y_means, [&mean_output](const auto& col) { return col - mean_output; });
      std::cout << "y_deviations" << std::endl << y_deviations << std::endl << std::flush;
      //
      return std::tuple {std::move(mean_output), std::move(x_deviations), std::move(y_deviations)};
    }

  public:
    template<typename InputDist, typename ... NoiseDist>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      std::cout << "------------------------------------------------------------------------" << std::endl;
      auto [mean_output, x_deviations, y_deviations] = trans(in, n...);
      auto [out_covariance, cross_covariance] = SamplePointsType::template covariance<InputDist, NoiseDist...>(x_deviations, y_deviations);
      auto out = GaussianDistribution {mean_output, out_covariance};
      return std::tuple {std::move(out), std::move(cross_covariance)};
    }

  };


  template<typename SamplePointsType, typename TransformationType>
  auto make_SamplePointsTransform(TransformationType&& f)
  {
    return SamplePointsTransform<SamplePointsType, std::decay_t<TransformationType>>(std::forward<TransformationType>(f));
  };


}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
