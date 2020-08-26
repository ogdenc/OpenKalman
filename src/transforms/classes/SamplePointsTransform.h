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
    template<typename Transformation, typename...Points, typename...Dists, std::size_t...ints>
    static constexpr auto y_means_impl(
      const Transformation& g,
      const std::tuple<Points...>& points,
      const std::tuple<Dists...>& dists,
      std::index_sequence<ints...>)
    {
      constexpr auto count = MatrixTraits<decltype(std::get<0>(points))>::columns;
      return apply_columnwise<count>([&](size_t i) {
        return to_Euclidean(g((column(std::get<ints>(points), i) + make_Matrix(mean(std::get<ints>(dists))))...));
      });
    }

  protected:
    template<std::size_t dim, typename InputDist, typename Transformation, typename XPointsTup, typename XDistsTup>
    static constexpr auto transform_impl(
      const Transformation& g,
      const XPointsTup& xpoints_tup,
      const XDistsTup& xdists_tup)
    {
      constexpr std::size_t N = std::tuple_size_v<XPointsTup>;
      static_assert(N == std::tuple_size_v<XDistsTup>);
      auto ymeans = y_means_impl(g, xpoints_tup, xdists_tup, std::make_index_sequence<N>());
      auto y_mean = SamplePointsType::template weighted_means<dim>(ymeans);
      auto xpoints0 = std::get<0>(xpoints_tup);
      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints = apply_columnwise(ymeans, [&y_mean](const auto& col) { return col - y_mean; });
      auto [y_covariance, cross_covariance] = SamplePointsType::template covariance<dim, InputDist>(xpoints0, ypoints);
      auto y = GaussianDistribution {y_mean, y_covariance};
      return std::tuple {ypoints, y, cross_covariance};
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
      const auto points_tuple = SamplePointsType::sample_points(x, n...);
      constexpr auto dim = (DistributionTraits<InputDist>::dimension + ... + DistributionTraits<NoiseDist>::dimension);
      //
      auto xdists_tup = std::forward_as_tuple(x, n...);
      auto [_, y, cross_covariance] = transform_impl<dim, InputDist>(g, points_tuple, xdists_tup);

      return std::tuple {std::move(y), std::move(cross_covariance)};
    }

    /**
     * Perform two sample points transforms.
     * @tparam Transformation1 The transformation on which the first transform is based.
     * @tparam Transformation2 The transformation on which the second transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distribution.
     **/
    template<
      typename InputDist, typename Transformation1, typename...QNoise, typename Transformation2, typename...RNoise,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>,
        is_distribution<QNoise>..., is_distribution<RNoise>...>, int> = 0>
    auto operator()(
      const InputDist& x,
      const std::tuple<Transformation1, QNoise...>& t1,
      const std::tuple<Transformation2, RNoise...>& t2) const
    {
      // The scaled sample points, divided into tuples for the input and each noise term:
      const auto qs = internal::tuple_slice<1, 1 + sizeof...(QNoise)>(t1);
      const auto rs = internal::tuple_slice<1, 1 + sizeof...(RNoise)>(t2);
      const auto points_tuple = std::apply([](const auto&...args) {
        return SamplePointsType::sample_points(args...);
      }, std::tuple_cat(std::tuple {x}, qs, rs));
      constexpr auto dim = ((DistributionTraits<InputDist>::dimension + ... + DistributionTraits<QNoise>::dimension) +
        ... + DistributionTraits<RNoise>::dimension);
      auto xpoints1 = std::get<0>(points_tuple);
      auto xpoints2 = internal::tuple_slice<1, 1 + sizeof...(QNoise)>(points_tuple);
      auto xpoints3 = internal::tuple_slice<1 + sizeof...(QNoise), 1 + sizeof...(QNoise) + sizeof...(RNoise)>(points_tuple);

      // First transform:
      decltype(auto) g1 = std::get<0>(t1);
      auto xpoints_tup = std::tuple_cat(std::tuple {xpoints1}, xpoints2);
      auto xdists_tup = std::tuple_cat(std::tuple {x}, qs);
      auto [ypoints, y, _] = transform_impl<dim, InputDist>(g1, xpoints_tup, xdists_tup);

      // Second transform:
      decltype(auto) g2 = std::get<0>(t2);
      auto ypoints_tup = std::tuple_cat(std::tuple {ypoints}, xpoints3);
      auto ydists_tup = std::tuple_cat(std::tuple {y}, rs);
      auto [zpoints, z, cross_covariance] = transform_impl<dim, InputDist>(g2, ypoints_tup, ydists_tup);

      return std::tuple {std::move(z), std::move(cross_covariance)};
    }

  };


}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
