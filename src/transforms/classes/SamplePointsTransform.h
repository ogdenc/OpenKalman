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

    explicit SamplePointsTransform(TransformationType&& transformation)
      : transformation(std::move(transformation)) {}

    const TransformationType transformation;

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
      const auto x_means_tuple = SamplePointsType::sample_points(x, n...);
      constexpr auto dim = (DistributionTraits<Dist>::dimension + ... + DistributionTraits<Noise>::dimension);
      constexpr auto sample_points_count = MatrixTraits<std::tuple_element_t<0, decltype(x_means_tuple)>>::columns;
      //
      const auto x_deviations = apply_columnwise(std::get<0>(x_means_tuple), [&x](const auto& col) { return col - mean(x); }); ///@TODO: We can eliminate this subtraction
      const auto y_means = apply_columnwise<sample_points_count>([&x_means_tuple, this](size_t i) {
        return std::apply([i, this](const auto&...input_terms) { return transformation(column(input_terms, i)...); },
        x_means_tuple);
      });
      //
      const auto mean_output = strict(SamplePointsType::template weighted_means<dim>(y_means));
      // Each column is a deviation from y mean for each transformed sigma point:
      const auto y_deviations = apply_columnwise(y_means, [&mean_output](const auto& col) {
        return col - mean_output; });
      //
      return std::tuple {mean_output, x_deviations, y_deviations};
    }

  public:
    template<typename InputDist, typename ... NoiseDist,
      std::enable_if_t<not std::disjunction_v<
        is_Cholesky<typename DistributionTraits<InputDist>::Covariance>,
        is_Cholesky<typename DistributionTraits<NoiseDist>::Covariance>...>, int> = 0>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      const auto [mean_output, x_deviations, y_deviations] = trans(in, n...);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      const auto [out_covariance, cross_covariance] = SamplePointsType::template covariance<dim>(x_deviations, y_deviations);
      const auto out = DistributionTraits<InputDist>::make(mean_output, out_covariance);
      return std::tuple {out, cross_covariance};
    }

    template<typename InputDist, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<
        is_Cholesky<typename DistributionTraits<InputDist>::Covariance>,
        is_Cholesky<typename DistributionTraits<NoiseDist>::Covariance>...>, int> = 0>
    auto operator()(const InputDist& in, const NoiseDist& ...n) const
    {
      const auto [mean_output, x_deviations, y_deviations] = trans(in, n...);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      const auto [out_covariance, cross_covariance] = SamplePointsType::template sqrt_covariance<dim>(x_deviations, y_deviations);
      const auto out = DistributionTraits<InputDist>::make(mean_output, out_covariance);
      return std::tuple {out, cross_covariance};
    }

  };


  template<typename SamplePointsType, typename TransformationType>
  auto make_SamplePointsTransform(TransformationType&& f)
  {
    return SamplePointsTransform<SamplePointsType, std::decay_t<TransformationType>>(std::forward<TransformationType>(f));
  };


}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
