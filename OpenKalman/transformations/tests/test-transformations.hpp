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


namespace OpenKalman::test
{
  using namespace OpenKalman;

  template<int n>
  inline auto sum_of_squares = Transformation
    {
      [](const auto& x, const auto& ...ps) // function
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        return sum(((transpose(x) * x), ... , ps));
      },
      [](const auto& x, const auto& ...ps) // Jacobians
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        return std::tuple_cat(
          std::tuple {2 * transpose(x)},
          OpenKalman::collections::internal::repeat_tuple_view<sizeof...(ps), decltype(Mean {1.})>(Mean {1.}));
      },
      [](const auto& x, const auto& ...ps) // Hessians
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        std::array<Matrix<Dimensions<n>, Dimensions<n>, eigen_matrix_t<double, n, n>>, 1> I;
        I[0] = 2 * eigen_matrix_t<double, n, n>::Identity();
        return std::tuple {std::move(I), std::get<0>(zero_hessian<Axis>(ps))...};
      }
    };

  template<int n>
  inline auto time_of_arrival = Transformation
    {
      [](const auto& x, const auto& ...ps) // function
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        return sum(apply_coefficientwise([](const auto& c) { return std::sqrt(c); }, adjoint(x) * x), ps...);
      },
      [](const auto& x, const auto& ...ps) // Jacobians
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        return std::make_tuple(scalar_quotient(adjoint(x), std::sqrt(trace(adjoint(x) * x))),
          Matrix<Axis, Axis, dense_writable_matrix_t<decltype(ps), Layout::none, scalar_type_of_t<decltype(ps)>, std::tuple<Axis, Axis>>> {1.}...);
      },
      [](const auto& x, const auto& ...ps) // Hessians
      {
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 0>, Dimensions<n>>);
        static_assert(compares_with<vector_space_descriptor_of_t<decltype(x), 1>, Axis >);
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 0>, Axis >and ...));
        static_assert((compares_with<vector_space_descriptor_of_t<decltype(ps), 1>, Axis >and ...));

        std::array<Matrix<Dimensions<n>, Dimensions<n>, eigen_matrix_t<double, n, n>>, 1> ret;
        double sq = trace(adjoint(x) * x);
        ret[0] = pow(sq, -1.5) * (-x * adjoint(x) + sq * make_identity_matrix_like<decltype(ret[0])>());
        return std::make_tuple(std::move(ret), std::get<0>(zero_hessian<Axis>(ps))...);
      }
    };

  namespace
  {
    using C2t = Dimensions<2>;
    using M2t = Mean<C2t, eigen_matrix_t < double, 2, 1>>;
    using M22t = Matrix <C2t, C2t, eigen_matrix_t<double, 2, 2>>;
  }

  inline auto radar = Transformation
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
          OpenKalman::collections::internal::repeat_tuple_view<sizeof...(ps), decltype(make_identity_matrix_like(ret))>(make_identity_matrix_like(ret)));
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
  using MP2t = Matrix<Polar<>, Dimensions<2>, eigen_matrix_t<double, 2, 2>>;
  using M2Pt = Matrix<Dimensions<2>, Polar<>, eigen_matrix_t<double, 2, 2>>;
  using MPPt = Matrix<Polar<>, Polar<>, eigen_matrix_t<double, 2, 2>>;

  inline auto radarP = Transformation
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
          std::move(ret)}, OpenKalman::collections::internal::repeat_tuple_view<sizeof...(ps), decltype(make_identity_matrix_like(ret))>(make_identity_matrix_like(ret)));
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

  inline auto Cartesian2polar = Transformation
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
          std::move(ret)}, OpenKalman::collections::internal::repeat_tuple_view<sizeof...(ps), decltype(make_identity_matrix_like<MPPt>())>(make_identity_matrix_like<MPPt>()));
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

  namespace
  {
    using Cyl = std::tuple<Polar<>, Axis>;
    using MC1t = Matrix <Cyl, Axis, eigen_matrix_t<double, 3, 1>>;
    using MS1t = Matrix <Spherical<>, Axis, eigen_matrix_t<double, 3, 1>>;
    using MSCt = Matrix <Spherical<>, Cyl, eigen_matrix_t<double, 3, 3>>;
    using MSSt = Matrix <Spherical<>, Spherical<>, eigen_matrix_t<double, 3, 3>>;
    using MCCt = Matrix <Cyl, Cyl, eigen_matrix_t<double, 3, 3>>;
  }

  inline auto Cylindrical2spherical = Transformation
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
          std::move(ret)}, OpenKalman::collections::internal::repeat_tuple_view<sizeof...(ps), decltype(make_identity_matrix_like<MSSt>())>(make_identity_matrix_like<MSSt>()));
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
        ret[1] = make_zero<MCCt>();
        ret[2] = {x02 * h42, 0, h2 - x00 * h42,
                  0, 0, 0,
                  x22 * h42 - h2, 0, -x02 * h42};
        return std::tuple {std::move(ret), std::get<0>(zero_hessian<Spherical<>>(ps))...};
      }
    };

} // namespace OpenKalman::test

#endif //OPENKALMAN_TESTS_TRANSFORMATIONS_HPP
