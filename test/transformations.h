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

template<int n>
inline const auto sum_of_squares = make_Transformation
  (
    [](const auto& x, const auto&...ps) // function
    {
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis>...>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis>...>);

      return strict(((transpose(x) * x) + ... + ps));
    },
    [](const auto& x, const auto&...ps) // Jacobians
    {
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis>...>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis>...>);

      return std::tuple {2 * transpose(x), internal::tuple_replicate<sizeof...(ps)>(Mean {1.})};
    },
    [](const auto& x, const auto&...ps) // Hessians
    {
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis>...>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis>...>);

      std::array<TypedMatrix<Axes<n>, Axes<n>, Eigen::Matrix<double, n, n>>, 1> I;
      I[0] = 2 * Eigen::Matrix<double, n, n>::Identity();

      return std::tuple_cat(std::tuple {I}, internal::tuple_replicate<sizeof...(ps)>(zero_hessian<Axis, decltype(x), decltype(ps)...>()));
    }
  );

template<int n>
inline const auto time_of_arrival = make_Transformation
  (
    [](const auto& x, const auto&...ps) // function
    {
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis>...>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis>...>);

      return strict((apply_coefficientwise(adjoint(x) * x, [](const auto& c) { return std::sqrt(c); }) + ... + ps));
    },
    [](const auto& x, const auto&...ps) // Jacobians
    {
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
      static_assert(is_equivalent_v<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis>...>);
      static_assert(std::conjunction_v<is_equivalent<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis>...>);

      return std::tuple_cat(std::tuple {strict(adjoint(x) / std::sqrt((x.adjoint() * x)(0,0)))},
        internal::tuple_replicate<sizeof...(ps)>(Mean {1.}));
    },
    [](const auto& x, const auto&...ps) // Hessians
    {
      std::array<TypedMatrix<Axes<n>, Axes<n>, Eigen::Matrix<double, n, n>>, 1> ret;
      double sq = (adjoint(x) * x)(0,0);
      ret[0] = pow(sq, -1.5) * (-x * adjoint(x) + sq * MatrixTraits<decltype(ret[0])>::identity());
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(zero_hessian<Axis, decltype(x), decltype(ps)...>()));
    }
  );

using C2t = Coefficients<Axis, Axis>;
using M2t = Mean<C2t, Eigen::Matrix<double, 2, 1>>;
using M22t = TypedMatrix<C2t, C2t, Eigen::Matrix<double, 2, 2>>;

inline const auto radar = make_Transformation
  (
    [](const M2t& x, const auto&...ps) // function
    {
      return (M2t {x(0) * cos(x(1)), x(0) * sin(x(1))} + ... + ps);
    },
    [](const M2t& x, const auto&...ps) // Jacobians
    {
      M22t ret = {
        std::cos(x(1)), -x(0) * std::sin(x(1)),
        std::sin(x(1)), x(0) * std::cos(x(1))};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(M2t::identity()));
    },
    [](const M2t& x, const auto&...ps) // Hessians
    {
      std::array<M22t, 2> ret;
      ret[0] = {0, -sin(x(1)),
                -sin(x(1)), -x(0) * cos(x(1))};
      ret[1] = {0, cos(x(1)),
                cos(x(1)), -x(0) * sin(x(1))};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(
        zero_hessian<C2t, decltype(x), decltype(ps)...>()));
    }
  );


using M2Pt = Mean<Polar<>, Eigen::Matrix<double, 2, 1>>;
using M22Pt = TypedMatrix<Polar<>, Axes<2>, Eigen::Matrix<double, 2, 2>>;
using M22PPt = TypedMatrix<Polar<>, Polar<>, Eigen::Matrix<double, 2, 2>>;

inline const auto radarP = make_Transformation
  (
    [](const M2Pt& x, const auto&...ps) // function
    {
      return (M2t {x(0) * cos(x(1)), x(0) * sin(x(1))} + ... + ps);
    },
    [](const M2Pt& x, const auto&...ps) // Jacobians
    {
      M22t ret = {
        std::cos(x(1)), -x(0) * std::sin(x(1)),
        std::sin(x(1)), x(0) * std::cos(x(1))};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(M22t::identity()));
    },
    [](const M2Pt& x, const auto&...ps) // Hessians
    {
      std::array<M22t, 2> ret;
      ret[0] = {0, -sin(x(1)),
                -sin(x(1)), -x(0) * cos(x(1))};
      ret[1] = {0, cos(x(1)),
                cos(x(1)), -x(0) * sin(x(1))};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(
        zero_hessian<C2t, decltype(x), decltype(ps)...>()));
    }
  );

inline const auto Cartesian2polar = make_Transformation
  (
    [](const M2t& x, const auto&...ps)
    {
      return (TypedMatrix {M2Pt {std::hypot(x(0), x(1)), std::atan2(x(1), x(0))}} + ... + TypedMatrix {ps});
    },
    [](const M2t& x, const auto&...ps) // Jacobians
    {
      const auto h = 1/std::hypot(x(1), x(0));
      M22Pt ret = {
        x(0)*h, x(1)*h,
        -x(1)*h*h, x(0)*h*h};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(M22PPt::identity()));
    },
    [](const M2t& x, const auto&...ps) // Hessians
    {
      const auto h = 1/std::hypot(x(1), x(0));
      std::array<M22Pt, 2> ret;
      ret[0] = {h - x(0)*x(0)*h*h*h, x(0)*x(1)*h*h*h,
                x(0)*x(1)*h*h*h, h - x(1)*x(1)*h*h*h};
      ret[1] = {2*x(0)*x(1)*h*h*h*h, -h*h + 2*x(0)*x(0)*h*h*h*h,
                -h*h + 2*x(1)*x(1)*h*h*h*h, 2*x(0)*x(1)*h*h*h*h};
      return std::tuple_cat(std::tuple {ret}, internal::tuple_replicate<sizeof...(ps)>(
        zero_hessian<Polar<>, decltype(x), decltype(ps)...>()));
    }
  );



#endif //OPENKALMAN_TESTS_TRANSFORMATIONS_H
