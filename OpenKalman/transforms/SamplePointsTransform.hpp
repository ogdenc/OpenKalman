/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of SamplePointsTransform and related aliases.
 */

#ifndef OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP
#define OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP

#include <functional>
#include <tuple>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;

  /**
   * \brief Scaled points transform. Compatible with unscented transform and cubature transform.
   * \details As implemented in, e.g.,
   * E. Wan & R. van der Merwe, "The unscented Kalman filter for nonlinear estimation,"
   * in Proc. of IEEE Symposium (AS-SPCC), pp. 153-158.
   * See also R. van der Merwe & E. Wan, "The Square-Root Unscented Kalman Filter for State and
   * Parameter-Estimation in Proc. Acoustics, Speech, and Signal Processing (ICASSP'01), 2001, pp. 3461-64.
   * \tparam SamplePointsType The type of sample points on which the transform is based (e.g., CubaturePoints).
   */
  template<typename SamplePointsType>
  struct SamplePointsTransform;


  using CubatureTransform = SamplePointsTransform<CubaturePoints>;

  using UnscentedTransform = SamplePointsTransform<UnscentedSigmaPoints>;
  using UnscentedTransformParameterEstimation = SamplePointsTransform<UnscentedSigmaPointsParameterEstimation>;


  template<typename SamplePointsType>
  struct SamplePointsTransform : oin::TransformBase<SamplePointsTransform<SamplePointsType>>
  {

  private:

    // Count the number of augmented input dimensions.
    template<typename D, typename...Ds>
    static constexpr auto count_dim(const std::tuple<Ds...>&)
    {
      return (DistributionTraits<D>::dimensions + ... + DistributionTraits<Ds>::dimensions);
    }


    // Reconstruct the flattened augmented sample points into a tuple of tuples.
    template<typename Ds, std::size_t pos = 0, std::size_t i = 0, typename FlattenedPs>
    static auto construct_ps(FlattenedPs&& flattened_ps)
    {
      if constexpr (i < std::tuple_size_v<Ds>)
      {
        constexpr auto group_size = std::tuple_size_v<std::decay_t<std::tuple_element_t<i, Ds>>>;
        auto ps_group = oin::tuple_slice<pos, pos + group_size>(std::forward<FlattenedPs>(flattened_ps));
        return std::tuple_cat(
          std::make_tuple(std::move(ps_group)),
          construct_ps<Ds, pos + group_size, i + 1>(std::forward<FlattenedPs>(flattened_ps)));
      }
      else // We hare reached the end of flattened_ps.
      {
        static_assert(pos == std::tuple_size_v<FlattenedPs>);
        return std::tuple {};
      }
    }


    template<std::size_t...ints, typename G, typename Dists, typename Points>
    static constexpr auto y_means_column(const G& g, Dists&& dists, Points&& points, std::size_t i)
    {
      return g((column(std::get<ints>(std::forward<Points>(points)), i) +
        mean_of(std::get<ints>(std::forward<Dists>(dists))))...);
    }


    template<typename G, typename Dists, typename Points, std::size_t...ints>
    static constexpr auto y_means_impl(const G& g, Dists&& dists, Points&& points, std::index_sequence<ints...>)
    {
      constexpr auto count = MatrixTraits<decltype(std::get<0>(points))>::columns;
      return apply_columnwise<count>([&](std::size_t i) {
        return y_means_column<ints...>(g, std::forward<Dists>(dists), std::forward<Points>(points), i);
      });
    }


    /**
     * \internal
     * \brief
     * \tparam dim The number of augmented sample points.
     * \tparam InputDist The initial input distribution.
     * \tparam return_cross Whether the transform must return a cross-covariance.
     * \tparam i The index of the current tests in Gs.
     * \tparam Gs A sequence of transformations.
     * \tparam D The input distribution for the current tests in <var>Gs</var>.
     * \tparam Ds A set comprising, for each tests in <var>Gs</var>, a tuple of zero or more noise
     * distributions that correspond to that tests.
     * \tparam P The augmented sample points corresponding to the current tests in <var>Gs</var>.
     * \tparam Ps The augmented sample points, as a tuple of tuples following the same structure as Ds.
     * \return The posterior covariance or, if <code>return_cross</cross> is <code>true</code>, a tuple containing
     * the posterior covariance and the cross-covariance.
     */
#ifdef __cpp_concepts
    template<std::size_t dim, typename InputDist, bool return_cross, std::size_t i = 0,
      typename Gs, typename D, typename Ds, typename P, typename Ps> requires
        (std::tuple_size_v<std::decay_t<Gs>> == std::tuple_size_v<std::decay_t<Ds>>) and
        (std::tuple_size_v<std::decay_t<Gs>> == std::tuple_size_v<std::decay_t<Ps>>)
#else
    template<std::size_t dim, typename InputDist, bool return_cross, std::size_t i = 0,
      typename Gs, typename D, typename Ds, typename P, typename Ps, std::enable_if_t<
        (std::tuple_size_v<std::decay_t<Gs>> == std::tuple_size_v<std::decay_t<Ds>>) and
        (std::tuple_size_v<std::decay_t<Gs>> == std::tuple_size_v<std::decay_t<Ps>>), int> = 0>
#endif
    static auto transform_impl(Gs&& gs, D&& d, Ds&& ds, P&& p, Ps&& ps)
    {
      auto g = std::get<i>(std::forward<Gs>(gs));
      auto xdists_tup = std::tuple_cat(std::make_tuple(std::forward<D>(d)), std::get<i>(std::forward<Ds>(ds)));
      auto xpoints_tup = std::tuple_cat(std::forward_as_tuple(p), std::get<i>(std::forward<Ps>(ps)));

      constexpr std::size_t N = std::tuple_size_v<decltype(xpoints_tup)>;
      static_assert(N == std::tuple_size_v<decltype(xdists_tup)>);

      auto y_means = Mean {
        y_means_impl(g, std::move(xdists_tup), std::move(xpoints_tup), std::make_index_sequence<N>())};

      const auto y_mean = from_euclidean(SamplePointsType::template weighted_means<dim>(to_euclidean(y_means)));

      // Each column is a deviation from y mean for each transformed sigma point:
      auto ypoints = apply_columnwise(
        std::move(y_means), [&](const auto& col) { return make_self_contained(col - y_mean);});

      if constexpr (i + 1 < std::tuple_size_v<std::decay_t<Gs>>)
      {
        auto y_covariance = SamplePointsType::template covariance<dim, InputDist, false>(std::forward<P>(p), ypoints);
        auto y = GaussianDistribution {make_self_contained(std::move(y_mean)), std::move(y_covariance)};
        return transform_impl<dim, InputDist, return_cross, i + 1>(
          std::forward<Gs>(gs), std::move(y), std::forward<Ds>(ds), std::move(ypoints), std::forward<Ps>(ps));
      }
      else // processing for the last transformation in Gs:
      {
        auto y_covariance = SamplePointsType::template covariance<dim, InputDist, return_cross>(
          std::forward<P>(p), std::move(ypoints));
        if constexpr (return_cross)
        {
          auto [y_cov, cross] = std::move(y_covariance);
          return std::tuple {GaussianDistribution {make_self_contained(std::move(y_mean)), std::move(y_cov)},
                             std::move(cross)};
        }
        else
        {
          return GaussianDistribution {make_self_contained(std::move(y_mean)), std::move(y_covariance)};
        }
      }
    }


    /**
     * \internal
     * \brief Perform one or more consecutive sample points transforms.
     * \tparam return_cross Whether the transform must return a cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Ts A list of tuples, each containing (1) a transformation and (2) zero or more noise terms for that
     * transformation.
     * \return The posterior distribution or, if <var>return_cross</var> is <code>true</code>, a tuple containing
     * the posterior distribution and the cross-covariance.
     **/
    template<bool return_cross, typename InputDist, typename...Ts>
    auto transform(InputDist&& x, Ts&&...ts) const
    {
      // Extract the transformations.
      auto gs = std::make_tuple(std::get<0>(std::forward<Ts>(ts))...);

      // Extract the noise terms.
      auto ds = std::make_tuple(oin::tuple_slice<1, std::tuple_size_v<std::decay_t<Ts>>>(std::forward<Ts>(ts))...);

      // Flatten ds to a 1D tuple of noise terms.
      auto flattened_ds = std::apply([](auto&&...args) {
        return std::tuple_cat(std::forward<decltype(args)>(args)...); }, std::move(ds));

      // Create a tuple comprising the augmented sample points (based on input x and each noise term in flattened_ds):
      auto points_tuple = std::apply([&x](auto&&...args) {
        return SamplePointsType::sample_points(x, std::forward<decltype(args)>(args)...);
      }, std::move(flattened_ds));

      // The augmented sample points corresponding to input x.
      auto p = std::get<0>(std::move(points_tuple));

      // The augmented sample points corresponding to each noise term in flattened_ds.
      auto flattened_ps = oin::tuple_slice<1, std::tuple_size_v<decltype(points_tuple)>>(std::move(points_tuple));

      // The augmented sample points, reconstructed into the same tuple-of-tuples structure as ds.
      auto ps = construct_ps<decltype(ds)>(std::move(flattened_ps));

      // Number of augmented input dimensions.
      constexpr auto dim = count_dim<InputDist>(flattened_ds);

      return SamplePointsTransform::transform_impl<dim, InputDist, return_cross>(
        std::move(gs), std::forward<InputDist>(x), std::move(ds), std::move(p), std::move(ps));
    }

  public:

    /**
     * \brief Perform one or more consecutive sample points transforms.
     * \tparam InputDist The prior distribution.
     * \tparam Ts A list of tuple-like structures, each containing arguments to a transform.
     * These arguments each include a tests and zero or more noise distributions.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, oin::tuple_like...Ts>
#else
    template<typename InputDist, typename...Ts, std::enable_if_t<
      gaussian_distribution<InputDist> and (oin::tuple_like<Ts> and ...), int> = 0>
#endif
    auto operator()(InputDist&& x, Ts&&...ts) const
    {
      return transform<false>(std::forward<InputDist>(x), std::forward<Ts>(ts)...);
    }


    /**
     * \brief Perform one sample points transform.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The tests on which the transform is based.
     * \tparam NoiseDist Zero or more noise distributions.
     * \return The posterior distribution.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, typename Trans, gaussian_distribution ... NoiseDists> requires
      requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto operator()(InputDist&& x, Trans&& g, NoiseDists&&...n) const
    {
      return transform<false>(
        std::forward<InputDist>(x), std::forward_as_tuple(std::forward<Trans>(g), std::forward<NoiseDists>(n)...));
    }


    /**
     * \brief Perform one or more consecutive sample points transforms, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Ts A list of tuple-like structures, each containing arguments to a transform.
     * These arguments each include a tests and zero or more noise distributions.
     * \return A tuple comprising the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, oin::tuple_like...Ts>
#else
    template<typename InputDist, typename...Ts, std::enable_if_t<
      gaussian_distribution<InputDist> and (oin::tuple_like<Ts> and ...), int> = 0>
#endif
    auto transform_with_cross_covariance(InputDist&& x, Ts&&...ts) const
    {
      return transform<true>(std::forward<InputDist>(x), std::forward<Ts>(ts)...);
    }


    /**
     * \brief Perform one sample points transform, also returning the cross-covariance.
     * \tparam InputDist The prior distribution.
     * \tparam Trans The tests on which the transform is based.
     * \tparam NoiseDist Zero or more noise distributions.
     * \return A tuple comprising the posterior distribution and the cross-covariance.
     **/
#ifdef __cpp_concepts
    template<gaussian_distribution InputDist, typename Trans, gaussian_distribution ... NoiseDists> requires
      requires(Trans g, InputDist x, NoiseDists...n) { g(mean_of(x), mean_of(n)...); }
#else
    template<typename InputDist, typename Trans, typename ... NoiseDists, std::enable_if_t<
      (gaussian_distribution<InputDist> and ... and gaussian_distribution<NoiseDists>) and
      std::is_invocable_v<Trans, typename DistributionTraits<InputDist>::Mean,
        typename DistributionTraits<NoiseDists>::Mean...>, int> = 0>
#endif
    auto transform_with_cross_covariance(InputDist&& x, Trans&& g, NoiseDists&& ...n) const
    {
      return transform<true>(
        std::forward<InputDist>(x), std::forward_as_tuple(std::forward<Trans>(g), std::forward<NoiseDists>(n)...));
    }

  };

}


#endif //OPENKALMAN_SAMPLEPOINTSTRANSFORM_HPP
