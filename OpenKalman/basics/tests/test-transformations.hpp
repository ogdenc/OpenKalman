/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TESTS_TRANSFORMATIONS_HPP
#define OPENKALMAN_TESTS_TRANSFORMATIONS_HPP

#include <array>
#include <iostream>


namespace OpenKalman::test
{
  using namespace OpenKalman;

  template<int n>
  inline const auto sum_of_squares = Transformation
    {
      [](const auto& x, const auto& ...ps) // function
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        return make_self_contained(((transpose(x) * x) + ... + ps));
      },
      [](const auto& x, const auto& ...ps) // Jacobians
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        return std::tuple_cat(std::tuple {
          2 * transpose(x)}, OpenKalman::internal::tuple_replicate<sizeof...(ps)>(Mean {1.}));
      },
      [](const auto& x, const auto& ...ps) // Hessians
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        std::array<Matrix<Axes<n>, Axes<n>, eigen_matrix_t<double, n, n>>, 1> I;
        I[0] = 2 * eigen_matrix_t<double, n, n>::Identity();
        return std::tuple {std::move(I), std::get<0>(zero_hessian<Axis>(ps))...};
      }
    };

  template<int n>
  inline const auto time_of_arrival = Transformation
    {
      [](const auto& x, const auto& ...ps) // function
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        return make_self_contained((apply_coefficientwise(adjoint(x) * x,
          [](const auto& c) { return std::sqrt(c); }) + ... + ps));
      },
      [](const auto& x, const auto& ...ps) // Jacobians
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        return std::tuple_cat(std::tuple {make_self_contained(adjoint(x) / std::sqrt(trace(adjoint(x) * x)))},
          OpenKalman::internal::tuple_replicate<sizeof...(ps)>(Mean {1.}));
      },
      [](const auto& x, const auto& ...ps) // Hessians
      {
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::RowCoefficients, Axes<n>>);
        static_assert(equivalent_to<typename MatrixTraits<decltype(x)>::ColumnCoefficients, Axis > );
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::RowCoefficients, Axis > and ...));
        static_assert((equivalent_to<typename MatrixTraits<decltype(ps)>::ColumnCoefficients, Axis > and ...));

        std::array<Matrix<Axes<n>, Axes<n>, eigen_matrix_t<double, n, n>>, 1> ret;
        double sq = trace(adjoint(x) * x);
        ret[0] = pow(sq, -1.5) * (-x * adjoint(x) + sq * MatrixTraits<decltype(ret[0])>::identity());
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<Axis>(ps))...};
      }
    };

  using C2t = Coefficients<Axis, Axis>;
  using M2t = Mean<C2t, eigen_matrix_t<double, 2, 1>>;
  using M22t = Matrix<C2t, C2t, eigen_matrix_t<double, 2, 2>>;

  inline const auto radar = Transformation
    {
      [](const M2t& x, const auto& ...ps) // function
      {
        return (M2t {x(0) * cos(x(1)), x(0) * sin(x(1))} + ... + ps);
      },
      [](const M2t& x, const auto& ...ps) // Jacobians
      {
        M22t ret = {
          std::cos(x(1)), -x(0) * std::sin(x(1)),
          std::sin(x(1)), x(0) * std::cos(x(1))};
        return std::tuple_cat(std::tuple {std::move(ret)},
          OpenKalman::internal::tuple_replicate<sizeof...(ps)>(M22t::identity()));
      },
      [](const M2t& x, const auto& ...ps) // Hessians
      {
        std::array<M22t, 2> ret;
        ret[0] = {0, -sin(x(1)),
                  -sin(x(1)), -x(0) * cos(x(1))};
        ret[1] = {0, cos(x(1)),
                  cos(x(1)), -x(0) * sin(x(1))};
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<C2t>(ps))...};
      }
    };


  using MP1t = Matrix<Polar<>, Axis, eigen_matrix_t<double, 2, 1>>;
  using MP2t = Matrix<Polar<>, Axes<2>, eigen_matrix_t<double, 2, 2>>;
  using M2Pt = Matrix<Axes<2>, Polar<>, eigen_matrix_t<double, 2, 2>>;
  using MPPt = Matrix<Polar<>, Polar<>, eigen_matrix_t<double, 2, 2>>;

  inline const auto radarP = Transformation
    {
      [](const MP1t& x, const auto& ...ps) // function
      {
        return (M2t {x(0) * cos(x(1)), x(0) * sin(x(1))} + ... + ps);
      },
      [](const MP1t& x, const auto& ...ps) // Jacobians
      {
        M2Pt ret = {
          std::cos(x(1)), -x(0) * std::sin(x(1)),
          std::sin(x(1)), x(0) * std::cos(x(1))};
        return std::tuple_cat(std::tuple {
          std::move(ret)}, OpenKalman::internal::tuple_replicate<sizeof...(ps)>(M22t::identity()));
      },
      [](const MP1t& x, const auto& ...ps) // Hessians
      {
        std::array<MPPt, 2> ret;
        ret[0] = {0, -sin(x(1)),
                  -sin(x(1)), -x(0) * cos(x(1))};
        ret[1] = {0, cos(x(1)),
                  cos(x(1)), -x(0) * sin(x(1))};
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<C2t>(ps))...};
      }
    };

  inline const auto Cartesian2polar = Transformation
    {
      [](const M2t& x, const auto& ...ps)
      {
        return (MP1t {std::hypot(x(0), x(1)), std::atan2(x(1), x(0))} + ... + Matrix {ps});
      },
      [](const M2t& x, const auto& ...ps) // Jacobians
      {
        const auto h = 1 / std::hypot(x(0), x(1));
        const auto h2 = h * h;
        MP2t ret = {
          x(0) * h, x(1) * h,
          -x(1) * h2, x(0) * h2};
        return std::tuple_cat(std::tuple {
          std::move(ret)}, OpenKalman::internal::tuple_replicate<sizeof...(ps)>(MPPt::identity()));
      },
      [](const M2t& x, const auto& ...ps) // Hessians
      {
        const auto h = 1 / std::hypot(x(0), x(1));
        const auto h2 = h * h;
        const auto h3 = h2 * h;
        const auto h42 = 2 * h2 * h2;
        const auto x00 = x(0) * x(0);
        const auto x01 = x(0) * x(1);
        const auto x11 = x(1) * x(1);
        std::array<M22t, 2> ret;
        ret[0] = {h - x00 * h3, -x01 * h3,
                  -x01 * h3, h - x11 * h3};
        ret[1] = {x01 * h42, h2 - x00 * h42,
                  x11 * h42 - h2, -x01 * h42};
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<Polar<>>(ps))...};
      }
    };

  using Cyl = Coefficients<Polar<>, Axis>;
  using MC1t = Matrix<Cyl, Axis, eigen_matrix_t<double, 3, 1>>;
  using MS1t = Matrix<Spherical<>, Axis, eigen_matrix_t<double, 3, 1>>;
  using MSCt = Matrix<Spherical<>, Cyl, eigen_matrix_t<double, 3, 3>>;
  using MSSt = Matrix<Spherical<>, Spherical<>, eigen_matrix_t<double, 3, 3>>;
  using MCCt = Matrix<Cyl, Cyl, eigen_matrix_t<double, 3, 3>>;

  inline const auto Cylindrical2spherical = Transformation
    {
      [](const MC1t& x, const auto& ...ps)
      {
        return (MS1t {std::hypot(x(0), x(2)), x(1), std::atan2(x(2), x(0))} + ... + Matrix {ps});
      },
      [](const MC1t& x, const auto& ...ps) // Jacobians
      {
        const auto h = 1 / std::hypot(x(0), x(2));
        const auto h2 = h * h;
        MSCt ret = {
          x(0) * h, 0, x(2) * h,
          0, 1, 0,
          -x(2) * h2, 0, x(0) * h2};
        return std::tuple_cat(std::tuple {
          std::move(ret)}, OpenKalman::internal::tuple_replicate<sizeof...(ps)>(MSSt::identity()));
      },
      [](const MC1t& x, const auto& ...ps) // Hessians
      {
        const auto h = 1 / std::hypot(x(0), x(2));
        const auto h2 = h * h;
        const auto h3 = h2 * h;
        const auto h42 = 2 * h2 * h2;
        const auto x00 = x(0) * x(0);
        const auto x02 = x(0) * x(2);
        const auto x22 = x(2) * x(2);
        std::array<MCCt, 3> ret;
        ret[0] = {h - x00 * h3, 0, -x02 * h3,
                  0, 0, 0,
                  -x02 * h3, 0, h - x22 * h3};
        ret[1] = MCCt::zero();
        ret[2] = {x02 * h42, 0, h2 - x00 * h42,
                  0, 0, 0,
                  x22 * h42 - h2, 0, -x02 * h42};
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<Spherical<>>(ps))...};
      }
    };

} // namespace OpenKalman::test

#endif //OPENKALMAN_TESTS_TRANSFORMATIONS_HPP
