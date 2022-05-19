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
 * A header file for the class KalmanFilter, relating to all Kalman-type recursive filters.
 */

#ifndef OPENKALMAN_KALMANFILTER_HPP
#define OPENKALMAN_KALMANFILTER_HPP

namespace OpenKalman
{
  /**
   * \brief A Kalman filter, using one or more statistical transforms.
   * \tparam Transforms Transforms for the filter.
   */
  template<typename...Transform>
  struct KalmanFilter;


  /**
   * \brief A Kalman filter, using the same transform for the process and the measurement.
   * \tparam Transform The transform for the physical process and the measurement.
   */
  template<typename Transform>
  struct KalmanFilter<Transform>
  {
  protected:
    Transform transform;

    template<typename XDistribution, typename YDistribution, typename CrossCovariance, typename Measurement>
    static auto
    update_step(const XDistribution& Nx, const YDistribution& Ny, const CrossCovariance& P_xy, const Measurement& z)
    {
      static_assert(gaussian_distribution<XDistribution>);
      static_assert(gaussian_distribution<YDistribution>);
      static_assert(typed_matrix<CrossCovariance>);
      static_assert(typed_matrix<Measurement> and column_vector<Measurement>);
      static_assert(equivalent_to<row_coefficient_types_of_t<Measurement>,
        typename DistributionTraits<YDistribution>::TypedIndex>);
      static_assert(equivalent_to<row_coefficient_types_of_t<CrossCovariance>,
        typename DistributionTraits<XDistribution>::TypedIndex>);
      static_assert(equivalent_to<column_coefficient_types_of_t<CrossCovariance>,
        typename DistributionTraits<YDistribution>::TypedIndex>);

      const auto y = mean_of(Ny);
      const auto P_yy = covariance_of(Ny);
      const auto K_adj = solve(adjoint(P_yy), adjoint(P_xy));
      const auto K = adjoint(K_adj);
      // K * P_yy == P_xy, or K == P_xy * inverse(P_yy)
      auto out_x_mean = make_self_contained(mean_of(Nx) + K * (Mean {z} - y));
      using re = typename DistributionTraits<XDistribution>::random_number_engine;

      if constexpr (cholesky_form<YDistribution>)
      {
        // P_xy * adjoint(K) == K * P_yy * adjoint(K) == K * square_root(P_yy) * adjoint(K * square_root(P_yy))
        // == square(LQ(K * square_root(P_yy)))
        auto out_x_cov = covariance_of(Nx) - square(LQ_decomposition(K * square_root(P_yy)));
        return make_GaussianDistribution<re>(std::move(out_x_mean), std::move(out_x_cov));
      }
      else
      {
        // K == P_xy * inverse(P_yy), so
        // P_xy * adjoint(K) == P_xy * adjoint(inverse(P_yy)) * adjoint(P_xy)
        auto out_x_cov = covariance_of(Nx) - Covariance(P_xy * K_adj);
        return make_GaussianDistribution<re>(std::move(out_x_mean), std::move(out_x_cov));
      }
    }

  public:
    explicit KalmanFilter(const Transform& trans)
      : transform(trans)
    {}

   /*
    * Predict the next state based on the process model.
    */
    template<typename...ProcessTransformArguments>
    auto
    predict(const ProcessTransformArguments&...args)
    {
      return transform(args...);
    }

   /*
    * Update the state based on a measurement and the measurement model.
    */
    template<typename Measurement, typename State, typename...MeasurementTransformArguments>
    auto
    update(const Measurement& z, const State& x, const MeasurementTransformArguments&...args)
    {
      const auto [y, P_xy] = transform.transform_with_cross_covariance(x, args...);
      return update_step(x, y, P_xy, z);
    }

   /*
    * Perform a complete predict-update cycle. Predict the next state based on the process model, and then
    * update the state based on a measurement and the measurement model.
    */
    template<
      typename Measurement,
      typename State,
      typename...ProcessTransformArgs,
      typename...MeasurementTransformArgs>
    auto
    operator()(
      const Measurement& z,
      const State& x,
      const std::tuple<ProcessTransformArgs...>& proc_args = std::tuple {},
      const std::tuple<MeasurementTransformArgs...>& meas_args = std::tuple {})
    {
      const auto&& [y_, P_xy, y] = std::apply(
        transform.transform_with_cross_covariance,
        std::tuple_cat(std::forward_as_tuple(x), proc_args, meas_args));
      return update_step(y, y_, P_xy, z);
    }

  };


  /**
   * \brief A Kalman filter, using a different statistical transform for the process and the measurement.
   * \tparam ProcessTransform The transform for the physical process.
   * \tparam MeasurementTransform An optional, separately-defined transform for the measurement.
   */
  template<typename ProcessTransform, typename MeasurementTransform>
  struct KalmanFilter<ProcessTransform, MeasurementTransform> : KalmanFilter<ProcessTransform>
  {
  protected:
    using Base = KalmanFilter<ProcessTransform>;
    using Base::transform;
    MeasurementTransform measurement_transform;

    using Base::update_step;

  public:
    KalmanFilter(const ProcessTransform& p_transform, const MeasurementTransform& m_transform)
      : Base(p_transform), measurement_transform(m_transform)
    {}

   /*
    * Update the state based on a measurement and the measurement model.
    */
    template<typename Measurement, typename State, typename...MeasurementTransformArguments>
    auto
    update(const Measurement& z, const State& x, const MeasurementTransformArguments&...args)
    {
      const auto [y, P_xy] = measurement_transform.transform_with_cross_covariance(x, args...);
      return update_step(x, y, P_xy, z);
    }

   /*
    * Perform a complete predict-update cycle. Predict the next state based on the process model, and then
    * update the state based on a measurement and the measurement model.
    */
    template<
      typename Measurement,
      typename State,
      typename...ProcessTransformArgs,
      typename...MeasurementTransformArgs>
    auto
    operator()(
      const Measurement& z,
      const State& x,
      const std::tuple<ProcessTransformArgs...>& proc_args = std::tuple {},
      const std::tuple<MeasurementTransformArgs...>& meas_args = std::tuple {})
    {
      const auto y = std::apply(transform, std::tuple_cat(std::forward_as_tuple(x), proc_args));
      const auto [y_, P_xy] = std::apply(
        measurement_transform.transform_with_cross_covariance,
        std::tuple_cat(std::forward_as_tuple(y), meas_args));
      return update_step(y, y_, P_xy, z);
    }

  };


  /////////////////////////////////////
  //        Deduction guides         //
  /////////////////////////////////////

  template<typename P>
  KalmanFilter(P&&) -> KalmanFilter<P>;

  template<typename P, typename M>
  KalmanFilter(P&&, M&&) -> KalmanFilter<P, M>;

}

#endif //OPENKALMAN_KALMANFILTER_HPP
