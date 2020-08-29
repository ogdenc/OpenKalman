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

    template<std::size_t pos = 0, typename FlattenedPs>
    static constexpr auto construct_ps(const FlattenedPs&)
    {
      static_assert(pos == std::tuple_size_v<FlattenedPs>);
      return std::tuple {};
    }

    template<std::size_t pos = 0, typename FlattenedPs, typename D, typename...Ds>
    static auto construct_ps(const FlattenedPs& flattened_ps, const D&, const Ds&...ds)
    {
      constexpr auto group_size = std::tuple_size_v<D>;
      auto ps_group = internal::tuple_slice<pos, pos + group_size>(flattened_ps);
      return std::tuple_cat(std::make_tuple(ps_group), construct_ps<pos + group_size>(flattened_ps, ds...));
    }

    template<typename D, typename...Ds>
    static constexpr auto count_dim(const std::tuple<Ds...>&)
    {
      return (DistributionTraits<D>::dimension + ... + DistributionTraits<Ds>::dimension);
    }

  protected:
    template<std::size_t dim, typename InputDist, std::size_t i,
      typename...Gs, typename P, typename...Ps, typename D, typename...Ds>
    static constexpr auto transform_impl(
      const std::tuple<Gs...>& gs,
      const P& xpoints,
      const std::tuple<Ps...>& ps,
      const D& x,
      const std::tuple<Ds...>& ds)
    {
      static_assert(sizeof...(Gs) == sizeof...(Ps) and sizeof...(Gs) == sizeof...(Ds));
      auto g = std::get<i>(gs);
      auto xpoints_tup = std::tuple_cat(std::tuple {xpoints}, std::get<i>(ps));
      auto xdists_tup = std::tuple_cat(std::tuple {x}, std::get<i>(ds));

      constexpr std::size_t N = std::tuple_size_v<decltype(xpoints_tup)>;
      static_assert(N == std::tuple_size_v<decltype(xdists_tup)>);

      auto ymeans = y_means_impl(g, xpoints_tup, xdists_tup, std::make_index_sequence<N>());
      auto y_mean = SamplePointsType::template weighted_means<dim>(ymeans);
      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints = apply_columnwise(ymeans, [&y_mean](const auto& col) { return col - y_mean; });
      auto [y_covariance, cross_covariance] = SamplePointsType::template covariance<dim, InputDist>(xpoints, ypoints);
      auto y = GaussianDistribution {y_mean, y_covariance};

      if constexpr(i + 1 < sizeof...(Gs))
      {
        return transform_impl<dim, InputDist, i + 1>(gs, ypoints, ps, y, ds);
      }
      else
      {
        return std::tuple {y, cross_covariance};
      }
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

      auto gs = std::tuple {g};
      auto ds = std::make_tuple(std::tuple {n...});
      auto xpoints = std::get<0>(points_tuple);
      auto ps = std::make_tuple(internal::tuple_slice<1, std::tuple_size_v<decltype(points_tuple)>>(points_tuple));
      return transform_impl<dim, InputDist, 0>(gs, xpoints, ps, x, ds);
    }

    /**
     * Perform one or more consecutive sample points transforms.
     * @tparam InputDist Input distribution.
     * @tparam Ts A list of tuples containing (1) a transformation and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename...Ts>
    auto operator()(const InputDist& x, const Ts&...ts) const
    {
      auto gs = std::tuple {std::get<0>(ts)...};
      auto ds = std::make_tuple(internal::tuple_slice<1, std::tuple_size_v<Ts>>(ts)...);

      auto flattened_ds = std::apply([](const auto&...args) {return std::tuple_cat(args...); }, ds);
      auto points_tuple = std::apply([&x](const auto&...args) {
        return SamplePointsType::sample_points(x, args...);
      }, flattened_ds);
      constexpr auto dim = count_dim<InputDist>(flattened_ds);

      auto xpoints = std::get<0>(points_tuple);
      auto flattened_ps = internal::tuple_slice<1, std::tuple_size_v<decltype(points_tuple)>>(points_tuple);
      auto ps = std::apply([&](const auto&...args) { return construct_ps(flattened_ps, args...); }, ds);

      return transform_impl<dim, InputDist, 0>(gs, xpoints, ps, x, ds);
    }

  };


}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_H
