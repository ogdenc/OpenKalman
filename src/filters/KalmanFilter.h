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


    template<typename DY, typename PxyType, typename Z>
    auto&
    kalmanUpdate(StateDist&& Dist_x, DY&& Dist_y, PxyType&& P_xy, Z&& z)
    {
      static_assert(is_covariance_v<typename DistributionTraits<DY>::Covariance>);
      static_assert(is_typed_matrix_v<PxyType>);
      static_assert(is_mean_v<Z>);
      static_assert(MatrixTraits<Z>::columns == 1);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::RowCoefficients, Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<PxyType>::ColumnCoefficients, typename DistributionTraits<DY>::Coefficients>);
      static_assert(OpenKalman::is_equivalent_v<typename MatrixTraits<Z>::RowCoefficients, typename DistributionTraits<DY>::Coefficients>);
      using CoeffsM = typename DistributionTraits<DY>::Coefficients;
      auto y = mean(std::forward<DY>(Dist_y));
      if constexpr (is_Cholesky_v<DY>)
      {
        auto S_yy = sqrt_covariance(std::forward<DY>(Dist_y));
        auto K = make_Matrix<Coefficients, CoeffsM>(adjoint(solve(adjoint(S_yy), adjoint(std::forward<PxyType>(P_xy)))));
        mean(Dist_x) += K * (std::forward<Z>(z) - std::move(y));
        covariance(Dist_x) -= std::move(K) * std::move(S_yy);
        // Note: this is equivalent to P_xx -= K * P_yy * K.adjoint() == K * S_yy * (K * S_yy).adjoint();
      }
      else
      {
        // Effectively, P_xx -= P_xy * adjoint(inverse(P_yy)) * adjoint(P_xy)
        auto P_yy = covariance(std::forward<DY>(Dist_y));
        auto K = make_Matrix<Coefficients, CoeffsM>(adjoint(solve(adjoint(std::move(P_yy)), adjoint(P_xy))));
        // Note: K == P_xy * inverse(P_yy)
        mean(Dist_x) += K * (std::forward<Z>(z) - std::move(y)); // == K(z - y)
        covariance(Dist_x) -= std::move(P_xy) * adjoint(std::move(K)); // == K * P_yy * adjoint(K)
      }
      return *this;
    }


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
                  return kalmanUpdate(ret, y, P_xy, z);
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
