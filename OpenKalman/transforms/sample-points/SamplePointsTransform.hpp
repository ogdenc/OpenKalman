/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP
#define OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP

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
  using UnscentedTransformParameterEstimation = SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>;


  template<typename SamplePointsType>
  struct SamplePointsTransform : TransformBase<SamplePointsTransform<SamplePointsType>>
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
        return g((column(std::get<ints>(points), i) + mean(std::get<ints>(dists)))...);
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
    template<std::size_t dim, typename InputDist, std::size_t i, bool return_cross = false,
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

      auto y_means = Mean {y_means_impl(g, xpoints_tup, xdists_tup, std::make_index_sequence<N>())};
      auto y_mean = from_Euclidean(SamplePointsType::template weighted_means<dim>(to_Euclidean(y_means)));
      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints = apply_columnwise(y_means, [&](const auto& col) { return strict(col - y_mean); });

      if constexpr (i + 1 < sizeof...(Gs))
      {
        auto y_covariance = SamplePointsType::template covariance<dim, InputDist, false>(xpoints, ypoints);
        auto y = GaussianDistribution {strict(std::move(y_mean)), std::move(y_covariance)};
        return transform_impl<dim, InputDist, i + 1, return_cross>(gs, ypoints, ps, y, ds);
      }
      else
      {
        auto y_covariance = SamplePointsType::template covariance<dim, InputDist, return_cross>(xpoints, ypoints);
        if constexpr (return_cross)
        {
          auto [y_cov, cross] = y_covariance;
          return std::tuple {GaussianDistribution {strict(std::move(y_mean)), std::move(y_cov)}, std::move(cross)};
        }
        else
        {
          return GaussianDistribution {std::move(y_mean), std::move(y_covariance)};
        }
      }
    }

    /**
     * Perform one or more consecutive sample points transforms.
     * @tparam InputDist Input distribution.
     * @tparam Ts A list of tuples containing (1) a transformation and (2) zero or more noise terms for that transformation.
     **/
    template<bool return_cross, typename InputDist, typename...Ts>
    auto transform(const InputDist& x, const Ts&...ts) const
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

      return transform_impl<dim, InputDist, 0, return_cross>(gs, xpoints, ps, x, ds);
    }

  public:
    /**
     * Perform one or more consecutive sample points transforms.
     * @tparam InputDist Input distribution.
     * @tparam Ts A list of tuples containing (1) a transformation and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename...T_args, typename...Ts, std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto operator()(const InputDist& x, const std::tuple<T_args...>& t, const Ts&...ts) const
    {
      return transform<false>(x, t, ts...);
    }

    /**
     * Perform one sample points transform.
     * @tparam Trans The transformation on which the transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distributions.
     **/
    template<typename InputDist, typename Trans, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...> and
        std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean, typename DistributionTraits<NoiseDist>::Mean...>, int> = 0>
    auto operator()(const InputDist& x, const Trans& g, const NoiseDist& ...n) const
    {
      return transform<false>(x, std::tuple {g, n...});
    }

    /**
     * Perform one or more consecutive sample points transforms, also returning the cross-covariance.
     * @tparam InputDist Input distribution.
     * @tparam Ts A list of tuples containing (1) a transformation and (2) zero or more noise terms for that transformation.
     **/
    template<typename InputDist, typename...T_args, typename...Ts, std::enable_if_t<is_distribution_v<InputDist>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const std::tuple<T_args...>& t, const Ts&...ts) const
    {
      return transform<true>(x, t, ts...);
    }

    /**
     * Perform one sample points transform, also returning the cross-covariance.
     * @tparam Trans The transformation on which the transform is based.
     * @tparam InputDist Input distribution.
     * @tparam NoiseDist Noise distributions.
     **/
    template<typename InputDist, typename Trans, typename ... NoiseDist,
      std::enable_if_t<std::conjunction_v<is_distribution<InputDist>, is_distribution<NoiseDist>...>, int> = 0>
    auto transform_with_cross_covariance(const InputDist& x, const Trans& g, const NoiseDist& ...n) const
    {
      return transform<true>(x, std::forward_as_tuple(g, n...));
    }

  };

}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP
