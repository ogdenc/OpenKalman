/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_TRANSFORMATIONS_H
#define OPENKALMAN_TESTS_TRANSFORMATIONS_H

#include <array>
#include <iostream>
#include <Eigen/Dense>

#include "transforms/transformations/Transformation.h"
#include "variables/coefficients/Angle.h"

using namespace OpenKalman;

//-------- Sum of squares --------//

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::none, 2> sum_of_squares
    {
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, 1> { return x.transpose() * x; },
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, n> { return 2 * x.transpose(); },
        [](Eigen::Matrix<Scalar, n, 1> x) -> std::array<Eigen::Matrix<Scalar, n, n>, 1>
        {
          std::array<Eigen::Matrix<Scalar, n, n>, 1> I;
          I[0] = 2 * Eigen::Matrix<Scalar, n, n>::Identity();
          return I;
        }
    };

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::additive, 2> sum_of_squares_add
    {
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, 1> { return x.transpose() * x; },
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, n> { return 2 * x.transpose(); },
        [](Eigen::Matrix<Scalar, n, 1> x) -> std::array<Eigen::Matrix<Scalar, n, n>, 1>
        {
          std::array<Eigen::Matrix<Scalar, n, n>, 1> I;
          I[0] = 2 * Eigen::Matrix<Scalar, n, n>::Identity();
          return I;
        }
    };

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::augmented, 2> sum_of_squares_aug
    {
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> Eigen::Matrix<Scalar, 1, 1>
        {
          Eigen::Matrix<Scalar, n, 1> x {x_aug.template head<n>()};
          return x.transpose() * x + Eigen::Matrix<Scalar, 1, 1> {x_aug(n)};
        },
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> Eigen::Matrix<Scalar, 1, n + 1>
        {
          Eigen::Matrix<Scalar, n, 1> x {x_aug.template head<n>()};
          Eigen::Matrix<Scalar, 1, n + 1> ret;
          ret << 2 * x.transpose(), 1;
          return ret;
        },
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> std::array<Eigen::Matrix<Scalar, n + 1, n + 1>, 1>
        {
          std::array<Eigen::Matrix<Scalar, n + 1, n + 1>, 1> I;
          I[0] = 2 * Eigen::Matrix<Scalar, n + 1, n + 1>::Identity();
          I[0](n, n) = 0;
          return I;
        }
    };


//-------- Time of arrival --------//

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::none, 2> time_of_arrival
    {
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, 1> { return (x.adjoint() * x).cwiseSqrt(); },
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, n>
        {
          return x.adjoint() / std::sqrt(x.adjoint() * x);
        },
        [](Eigen::Matrix<Scalar, n, 1> x) -> std::array<Eigen::Matrix<Scalar, n, n>, 1>
        {
          std::array<Eigen::Matrix<Scalar, n, n>, 1> ret;
          Scalar sq = x.adjoint() * x;
          ret[0] = pow(sq, -1.5) * (-x * x.adjoint() + sq * Eigen::Matrix<Scalar, n, n>::Identity());
          return ret;
        }
    };

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::additive, 2> time_of_arrival_add
    {
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, 1> { return (x.adjoint() * x).cwiseSqrt(); },
        [](Eigen::Matrix<Scalar, n, 1> x) -> Eigen::Matrix<Scalar, 1, n>
        {
          return x.adjoint() / std::sqrt(x.adjoint() * x);
        },
        [](Eigen::Matrix<Scalar, n, 1> x) -> std::array<Eigen::Matrix<Scalar, n, n>, 1>
        {
          std::array<Eigen::Matrix<Scalar, n, n>, 1> ret;
          Scalar sq = x.adjoint() * x;
          ret[0] = pow(sq, -1.5) * (-x * x.adjoint() + sq * Eigen::Matrix<Scalar, n, n>::Identity());
          return ret;
        }
    };

template<typename Scalar, int n>
static const Transformation<Scalar, Axes<n>, Coefficients<Axis>, NoiseType::augmented, 2> time_of_arrival_aug
    {
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> Eigen::Matrix<Scalar, 1, 1>
        {
          Eigen::Matrix<Scalar, n, 1> x {x_aug.template head<n>()};
          return (x.adjoint() * x).cwiseSqrt() + Eigen::Matrix<Scalar, 1, 1> {x_aug(n)};
        },
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> Eigen::Matrix<Scalar, 1, n + 1>
        {
          Eigen::Matrix<Scalar, n, 1> x {x_aug.template head<n>()};
          Eigen::Matrix<Scalar, 1, n + 1> ret;
          ret << x.adjoint() / std::sqrt(x.adjoint() * x), 1;
          return ret;
        },
        [](Eigen::Matrix<Scalar, n + 1, 1> x_aug) -> std::array<Eigen::Matrix<Scalar, n + 1, n + 1>, 1>
        {
          Eigen::Matrix<Scalar, n, 1> x {x_aug.template head<n>()};
          std::array<Eigen::Matrix<Scalar, n + 1, n + 1>, 1> ret;
          std::array<Eigen::Matrix<Scalar, n, n>, 1> ret_sub;
          Scalar sq = x.adjoint() * x;
          ret_sub = pow(sq, -1.5) * (-x * x.adjoint() + sq * Eigen::Matrix<Scalar, n, n>::Identity());
          ret[0] << ret_sub, Eigen::Matrix<Scalar, n, 1>::Zero(),
              Eigen::Matrix<Scalar, 1, n>::Zero(), Eigen::Matrix<Scalar, 1, 1>::Constant(0);
          return ret;
        }
    };


//-------- Radar --------//

using C2 = Coefficients<Axis, Axis>;

template<typename Scalar>
static const Transformation<Scalar, C2, C2, NoiseType::none, 2> radar
    {
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)), x(0) * sin(x(1))};
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 2>
        {
          Eigen::Matrix<Scalar, 2, 2> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          return ret;
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> std::array<Eigen::Matrix<Scalar, 2, 2>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 2, 2>, 2> ret;
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };

template<typename Scalar>
static const Transformation<Scalar, C2, C2, NoiseType::additive, 2> radar_add
    {
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)), x(0) * sin(x(1))};
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 2>
        {
          Eigen::Matrix<Scalar, 2, 2> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          return ret;
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> std::array<Eigen::Matrix<Scalar, 2, 2>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 2, 2>, 2> ret;
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };

template<typename Scalar>
static const Transformation<Scalar, C2, C2, NoiseType::augmented, 2> radar_aug
    {
        [](Eigen::Matrix<Scalar, 4, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)) + x(2), x(0) * sin(x(1)) + x(3)};
        },
        [](Eigen::Matrix<Scalar, 4, 1> x) -> Eigen::Matrix<Scalar, 2, 4>
        {
          Eigen::Matrix<Scalar, 2, 4> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(0, 2) = 1;
          ret(0, 3) = 0;
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          ret(1, 2) = 0;
          ret(1, 3) = 1;
          return ret;
        },
        [](Eigen::Matrix<Scalar, 4, 1> x) -> std::array<Eigen::Matrix<Scalar, 4, 4>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 4, 4>, 2> ret;
          ret[0] = Eigen::Matrix<Scalar, 4, 4>::Zero();
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1] = Eigen::Matrix<Scalar, 4, 4>::Zero();
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };

//-------- RadarPolar --------//

template<typename Scalar>
static const Transformation<Scalar, Coefficients<Polar>, C2, NoiseType::none, 2> radarP
    {
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)), x(0) * sin(x(1))};
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 2>
        {
          Eigen::Matrix<Scalar, 2, 2> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          return ret;
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> std::array<Eigen::Matrix<Scalar, 2, 2>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 2, 2>, 2> ret;
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };

template<typename Scalar>
static const Transformation<Scalar, Coefficients<Polar>, C2, NoiseType::additive, 2> radarP_add
    {
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)), x(0) * sin(x(1))};
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> Eigen::Matrix<Scalar, 2, 2>
        {
          Eigen::Matrix<Scalar, 2, 2> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          return ret;
        },
        [](Eigen::Matrix<Scalar, 2, 1> x) -> std::array<Eigen::Matrix<Scalar, 2, 2>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 2, 2>, 2> ret;
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };

template<typename Scalar>
static const Transformation<Scalar, Coefficients<Polar>, C2, NoiseType::augmented, 2> radarP_aug
    {
        [](Eigen::Matrix<Scalar, 4, 1> x) -> Eigen::Matrix<Scalar, 2, 1>
        {
          return Eigen::Matrix<Scalar, 2, 1> {x(0) * cos(x(1)) + x(2), x(0) * sin(x(1)) + x(3)};
        },
        [](Eigen::Matrix<Scalar, 4, 1> x) -> Eigen::Matrix<Scalar, 2, 4>
        {
          Eigen::Matrix<Scalar, 2, 4> ret;
          ret(0, 0) = cos(x(1));
          ret(0, 1) = -x(0) * sin(x(1));
          ret(0, 2) = 1;
          ret(0, 3) = 0;
          ret(1, 0) = sin(x(1));
          ret(1, 1) = x(0) * cos(x(1));
          ret(1, 2) = 0;
          ret(1, 3) = 1;
          return ret;
        },
        [](Eigen::Matrix<Scalar, 4, 1> x) -> std::array<Eigen::Matrix<Scalar, 4, 4>, 2>
        {
          std::array<Eigen::Matrix<Scalar, 4, 4>, 2> ret;
          ret[0] = Eigen::Matrix<Scalar, 4, 4>::Zero();
          ret[0](0, 0) = 0;
          ret[0](0, 1) = -sin(x(1));
          ret[0](1, 0) = -sin(x(1));
          ret[0](1, 1) = -x(0) * cos(x(1));
          ret[1] = Eigen::Matrix<Scalar, 4, 4>::Zero();
          ret[1](0, 0) = 0;
          ret[1](0, 1) = cos(x(1));
          ret[1](1, 0) = cos(x(1));
          ret[1](1, 1) = -x(0) * sin(x(1));
          return ret;
        }
    };


#endif //OPENKALMAN_TESTS_TRANSFORMATIONS_H
