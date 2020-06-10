/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_KALMANFILTER_H
#define OPENKALMAN_KALMANFILTER_H

#include <Eigen/Dense>
#include <iostream>
#include "utilities/NoiseType.h"

namespace OpenKalman
{
  /**
   * @brief A Kalman filter, using any statistical transform. Propagates either covariances or their square roots.
   * @tparam Dist The distribution type (e.g., Distribution).
   * @tparam Scalar The scalar type.
   * @tparam StateCoeffs Coefficients of the state vector.
   * @tparam MeasurementCoeffs Coefficients of the measurement vector.
   * @tparam Noise Noise parameters (state, measurement).
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs,
      typename ... Noise>
  struct KalmanFilter;

  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs,
      typename ... StateNoise,
      typename ... MeasurementNoise>
  struct KalmanFilter<Dist, Scalar, StateCoeffs, MeasurementCoeffs, std::tuple<StateNoise ...>, std::tuple<MeasurementNoise ...>>
  {
    using StateDist = Dist<Scalar, StateCoeffs>;
    using MeasurementDist = Dist<Scalar, MeasurementCoeffs>;
    using Measurement = Mean<Scalar, MeasurementCoeffs>;

    /**
     * @brief Construct Kalman filter, using separate process and measurement transforms
     * @param state_transform the transform to be used for the state prediction
     * @param measurement_transform the transform to be used for the measurement prediction
     */
    template<
        template<
        template<typename, typename> typename,
        typename,
        typename,
        typename, NoiseType,
        typename ...> typename ProcessTransform,
        NoiseType state_noise_t,
        template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...>
        typename MeasurementTransform,
        NoiseType measurement_noise_t>
    KalmanFilter(
        const ProcessTransform<Dist, Scalar, StateCoeffs, StateCoeffs, state_noise_t, StateNoise ...>& state_transform,
        const MeasurementTransform<Dist, Scalar, StateCoeffs, MeasurementCoeffs, measurement_noise_t, MeasurementNoise ...>& measurement_transform)
        :
        predict
            {
                [state_transform](const StateDist& x, const StateNoise& ... v) -> const StateDist
                {
                  const auto[y, _] = state_transform(x, v ...);
                  return y;
                }
            },
        update
            {
                [measurement_transform](
                    const StateDist& x,
                    const Measurement& z,
                    const MeasurementNoise& ... u) -> const StateDist
                {
                  const auto[y, P_xy] = measurement_transform(x, u ...);
                  StateDist ret {x};
                  return ret.kalmanUpdate(y, P_xy, z);
                }
            },
        predict_update
            {
                [this](
                    const StateDist& x,
                    const StateNoise& ... v,
                    const Measurement& z,
                    const MeasurementNoise& ... u) -> const StateDist
                {
                  return update(predict(x, v ...), z, u ...);
                }
            } {}


    /**
     * @brief Construct Kalman filter, using the same transform for both process and measurement
     * @param state_transform the transform to be used for the state prediction
     * @param measurement_transform the transform to be used for the measurement prediction
     */
    template<
        template<
        template<typename, typename> typename,
        typename,
        typename,
        typename, NoiseType,
        typename ...> typename Transform,
        NoiseType noise_t>
    explicit KalmanFilter(
        const Transform<Dist, Scalar, StateCoeffs, MeasurementCoeffs, noise_t, StateNoise ..., MeasurementNoise ...>& transform)
        :
        predict
            {
                [transform](const StateDist& x, const StateNoise& ... v) -> const StateDist
                {
                  const auto[y, _] = transform.first_half(x, v ...);
                  return y;
                }
            },
        update
            {
                [transform](const StateDist& x, const Measurement& z, const MeasurementNoise& ... u) -> const StateDist
                {
                  const auto[y, P_xy] = transform.second_half(x, u ...);
                  StateDist ret {x};
                  return ret.kalmanUpdate(y, P_xy, z);
                }
            },
        predict_update
            {
                [transform](
                    const StateDist& x,
                    const StateNoise& ... v,
                    const Measurement& z,
                    const MeasurementNoise& ... u) -> const StateDist
                {
                  const auto[y, P_xy] = transform(x, v ..., u ...);
                  StateDist ret {x};
                  return ret.kalmanUpdate(y, P_xy, z);
                }
            } {}

    /**
     * @brief Predict the next state distribution, using prior state distribution and process noise.
     * @param x Distribution of the current state.
     * @param v Process noise.
     * @return Updated distribution of the state.
     */
    const std::function<const StateDist(const StateDist& x, const StateNoise& ... v)>
        predict;

    /**
     * @brief Update the state distribution, using prior state distribution, a measurement, and measurement noise.
     * @param x Distribution of the current state.
     * @param z The measurement.
     * @param u Measurement noise.
     * @return Updated distribution of the state.
     */
    const std::function<const StateDist(const StateDist& x, const Measurement& z, const MeasurementNoise& ... u)>
        update;

    /**
     * @brief Update the state distribution, using prior state distribution, a measurement, and measurement noise.
     * @param x Distribution of the current state.
     * @param v Process noise.
     * @param z The measurement.
     * @param u Measurement noise.
     * @return Updated distribution of the state.
     */
    const std::function<const StateDist(
        const StateDist& x,
        const StateNoise& ... v,
        const Measurement& z,
        const MeasurementNoise& ... u)>
        predict_update;

  };


  /**
   * Deduction guides
   */
  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs,
      typename ... StateNoise,
      typename ... MeasurementNoise,
      template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...>
      typename ProcessTransform,
      NoiseType state_noise_t,
      template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...>
      typename MeasurementTransform,
      NoiseType measurement_noise_t>
  KalmanFilter(
      const ProcessTransform<Dist, Scalar, StateCoeffs, StateCoeffs, state_noise_t, StateNoise ...>&,
      const MeasurementTransform<Dist, Scalar, StateCoeffs, MeasurementCoeffs, measurement_noise_t, MeasurementNoise ...>&)
  ->
  KalmanFilter<Dist, Scalar, StateCoeffs, MeasurementCoeffs, std::tuple<StateNoise ...>, std::tuple<MeasurementNoise ...>>;

  template<
      template<typename, typename> typename Dist,
      typename Scalar,
      typename StateCoeffs,
      typename MeasurementCoeffs,
      typename StateNoise,
      typename MeasurementNoise,
      template<template<typename, typename> typename, typename, typename, typename, NoiseType, typename ...>
      typename Transform,
      NoiseType noise_t,
      typename = std::enable_if_t<noise_t != NoiseType::none>>
  KalmanFilter(
      const Transform<Dist, Scalar, StateCoeffs, MeasurementCoeffs, noise_t, StateNoise, MeasurementNoise>&)
  ->
  KalmanFilter<Dist, Scalar, StateCoeffs, MeasurementCoeffs, std::tuple<StateNoise>, std::tuple<MeasurementNoise>>;

}

#endif //OPENKALMAN_KALMANFILTER_H
