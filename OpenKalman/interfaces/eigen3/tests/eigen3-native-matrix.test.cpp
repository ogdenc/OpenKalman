/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"

using namespace OpenKalman;

using Mat2 = native_matrix_t<double, 2, 2>;
using Mat3 = native_matrix_t<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;

TEST_F(eigen3, Eigen_Matrix_class_traits)
{
  static_assert(eigen_native<Mat2>);
  static_assert(not eigen_native<double>);
  static_assert(eigen_matrix<Mat2>);
  static_assert(not eigen_matrix<double>);
  EXPECT_TRUE(is_near(MatrixTraits<native_matrix_t<double, 2, 3>>::zero(), make_native_matrix<double, 2, 3>(0, 0, 0, 0, 0, 0)));
  EXPECT_TRUE(is_near(MatrixTraits<Mat3>::identity(), make_native_matrix<Mat3>(1, 0, 0, 0, 1, 0, 0, 0, 1)));
  EXPECT_NEAR((make_native_matrix<double, 2, 2>(1, 2, 3, 4))(0, 0), 1, 1e-6);
  EXPECT_NEAR((make_native_matrix<double, 2, 2>(1, 2, 3, 4))(0, 1), 2, 1e-6);
  auto d1 = make_native_matrix<double, 3, 1>(1, 2, 3);
  EXPECT_NEAR(d1(1), 2, 1e-6);
  d1(0) = 5;
  d1(2, 0) = 7;
  EXPECT_TRUE(is_near(d1, make_native_matrix<double, 3, 1>(5, 2, 7)));
}

TEST_F(eigen3, Eigen_Matrix_overloads)
{
  EXPECT_TRUE(is_near(transpose(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6)));
  EXPECT_TRUE(is_near(adjoint(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 3, 2>(1, 4, 2, 5, 3, 6)));
  EXPECT_NEAR(determinant(make_native_matrix<double, 2, 2>(1, 2, 3, 4)), -2, 1e-6);
  EXPECT_NEAR(determinant(make_native_matrix<double, 1, 1>(2)), 2, 1e-6);
  EXPECT_NEAR(trace(make_native_matrix<double, 2, 2>(1, 2, 3, 4)), 5, 1e-6);
  EXPECT_NEAR(trace(make_native_matrix<double, 1, 1>(3)), 3, 1e-6);
  EXPECT_TRUE(is_near(solve(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 1>(5, 6)),
    make_native_matrix<double, 2, 1>(-4, 4.5)));
  EXPECT_TRUE(is_near(solve(make_native_matrix<double, 1, 1>(2),
    make_native_matrix<double, 1, 1>(6)),
    make_native_matrix<double, 1, 1>(3)));
  EXPECT_TRUE(is_near(reduce_columns(make_native_matrix<double, 2, 3>(1, 2, 3, 4, 5, 6)),
    make_native_matrix<double, 2, 1>(2, 5)));
  EXPECT_TRUE(is_near(LQ_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.08, 0.36, -1.640)),
    make_native_matrix<double, 2, 2>(-0.1, 0, 1.096, -1.272)));
  EXPECT_TRUE(is_near(QR_decomposition(make_native_matrix<double, 2, 2>(0.06, 0.36, 0.08, -1.640)),
    make_native_matrix<double, 2, 2>(-0.1, 1.096, 0, -1.272)));
  //
  using N = std::normal_distribution<double>::param_type;
  using Mat = native_matrix_t<double, 3, 1>;
  Mat m;
  m = randomize<Mat>(0.0, 1.0);
  m = randomize<Mat>(0.0, 0.7);
  m = Mat::Zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat>(N {1.0, 0.3})) / (i + 1);
  }
  auto offset1 = Mat::Constant(1);
  EXPECT_TRUE(is_near(m, offset1, 0.1));
  EXPECT_FALSE(is_near(m, offset1, 1e-8));

  Mat2 m22;
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<Mat2>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }
  auto offset2 = MatrixTraits<Mat2>::make(1., 1., 2., 2.);
  EXPECT_TRUE(is_near(m22, offset2, 0.1));
  EXPECT_FALSE(is_near(m22, offset2, 1e-8));

  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<Mat2>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }
  auto offset3 = MatrixTraits<Mat2>::make(1., 2., 3., 4.);
  EXPECT_TRUE(is_near(m22, offset3, 0.1));
  EXPECT_FALSE(is_near(m22, offset3, 1e-8));
}

TEST_F(eigen3, ZeroMatrix)
{
  EXPECT_NEAR((ZeroMatrix<native_matrix_t<double, 2, 2>>()(0, 0)), 0, 1e-6);
  EXPECT_NEAR((ZeroMatrix<native_matrix_t<double, 2, 2>>()(0, 1)), 0, 1e-6);
  EXPECT_TRUE(is_near(make_native_matrix(ZeroMatrix<native_matrix_t<double, 2, 3>>()), native_matrix_t<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(make_self_contained(ZeroMatrix<native_matrix_t<double, 2, 3>>()), native_matrix_t<double, 2, 3>::Zero()));
  EXPECT_NEAR(determinant(ZeroMatrix<native_matrix_t<double, 2, 2>>()), 0, 1e-6);
  EXPECT_NEAR(trace(ZeroMatrix<native_matrix_t<double, 2, 2>>()), 0, 1e-6);
  EXPECT_TRUE(is_near(reduce_columns(ZeroMatrix<native_matrix_t<double, 2, 3>>()), (native_matrix_t<double, 2, 1>::Zero())));
  EXPECT_NEAR(get_element(ZeroMatrix<native_matrix_t<double, 2, 2>>(), 1, 0), 0, 1e-8);
  static_assert(not element_settable<ZeroMatrix<native_matrix_t<double, 2, 2>>, 2>);
  EXPECT_TRUE(is_near(column(ZeroMatrix<native_matrix_t<double, 2, 3>>(), 1), (native_matrix_t<double, 2, 1>::Zero())));
  EXPECT_TRUE(is_near(column<1>(ZeroMatrix<native_matrix_t<double, 2, 3>>()), (native_matrix_t<double, 2, 1>::Zero())));
}

TEST_F(eigen3, Matrix_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 1, 2>(5, 6)), make_native_matrix<double, 3, 2>(1, 2, 3, 4, 5, 6)));
  EXPECT_TRUE(is_near(concatenate_horizontal(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 1>(5, 6)), make_native_matrix<double, 2, 3>(1, 2, 5, 3, 4, 6)));
  EXPECT_TRUE(is_near(concatenate_diagonal(make_native_matrix<double, 2, 2>(1, 2, 3, 4),
    make_native_matrix<double, 2, 2>(5, 6, 7, 8)), make_native_matrix<double, 4, 4>(1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8)));
  EXPECT_TRUE(is_near(split_vertical(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  native_matrix_t<double, 5, 3> x1;
  x1 <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3,
    4, 0, 0,
    0, 5, 0;
  auto a1 = split_vertical<3, 2>(x1);
  EXPECT_TRUE(is_near(a1, std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 2, 3>(
                 4, 0, 0,
                 0, 5, 0)}));
  auto a2 = split_vertical<3, 2>(make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      4, 0, 0,
      0, 5, 0));
  EXPECT_TRUE(is_near(a2,
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 2, 3>(
                 4, 0, 0,
                 0, 5, 0)}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>(make_native_matrix<double, 5, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      4, 0, 0,
      0, 5, 0)),
    std::tuple{make_native_matrix<double, 2, 3>(
      1, 0, 0,
      0, 2, 0),
    make_native_matrix<double, 2, 3>(
                 0, 0, 3,
                 4, 0, 0)}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(MatrixTraits<native_matrix_t<double, 5, 3>>::zero()),
    std::tuple {native_matrix_t<double, 3, 3>::Zero(), native_matrix_t<double, 2, 3>::Zero()}));
  EXPECT_TRUE(is_near(split_horizontal(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 3, 2>(
                 0, 0,
                 4, 0,
                 0, 5)}));
  const auto b1 = make_native_matrix<double, 3, 5>(
    1, 0, 0, 0, 0,
    0, 2, 0, 4, 0,
    0, 0, 3, 0, 5);
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(b1),
    std::tuple{make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3),
               make_native_matrix<double, 3, 2>(
                 0, 0,
                 4, 0,
                 0, 5)}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>(make_native_matrix<double, 3, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 2>(
      1, 0,
      0, 2,
      0, 0),
    make_native_matrix<double, 3, 2>(
                 0, 0,
                 0, 4,
                 3, 0)}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(MatrixTraits<native_matrix_t<double, 3, 5>>::zero()),
    std::tuple {native_matrix_t<double, 3, 3>::Zero(), native_matrix_t<double, 3, 2>::Zero()}));
  EXPECT_TRUE(is_near(split_diagonal(make_native_matrix<double, 2, 2>(1, 0, 0, 2)), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(make_native_matrix<double, 5, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5)),
    std::tuple{make_native_matrix<double, 3, 3>(
        1, 0, 0,
        0, 2, 0,
        0, 0, 3),
    make_native_matrix<double, 2, 2>(
                 4, 0,
                 0, 5)}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>(make_native_matrix<double, 5, 5>(
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5)),
    std::tuple{make_native_matrix<double, 2, 2>(
      1, 0,
      0, 2),
    make_native_matrix<double, 2, 2>(
                 3, 0,
                 0, 4)}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(MatrixTraits<native_matrix_t<double, 5, 5>>::zero()),
    std::tuple {native_matrix_t<double, 3, 3>::Zero(), native_matrix_t<double, 2, 2>::Zero()}));

  native_matrix_t<double, 2, 2> el; el << 1, 2, 3, 4;
  set_element(el, 5.5, 1, 0);
  EXPECT_NEAR(get_element(el, 1, 0), 5.5, 1e-8);
  static_assert(element_gettable<native_matrix_t<double, 3, 2>, 2>);
  static_assert(element_gettable<native_matrix_t<double, 3, 1>, 1>);
  static_assert(element_settable<native_matrix_t<double, 3, 2>, 2>);
  static_assert(element_settable<native_matrix_t<double, 3, 1>, 1>);
  static_assert(not element_gettable<native_matrix_t<double, 3, 2>, 1>);
  static_assert(element_gettable<native_matrix_t<double, 3, 1>, 2>);
  static_assert(not element_settable<native_matrix_t<double, 3, 2>, 1>);
  static_assert(element_settable<native_matrix_t<double, 3, 1>, 2>);
  static_assert(not element_settable<const native_matrix_t<double, 3, 2>, 2>);
  static_assert(not element_settable<const native_matrix_t<double, 3, 1>, 1>);

  EXPECT_TRUE(is_near(column(make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3), 2), make_native_matrix(0., 0, 3)));
  EXPECT_TRUE(is_near(column(MatrixTraits<native_matrix_t<double, 3, 3>>::zero(), 2), make_native_matrix(0., 0, 0)));
  EXPECT_TRUE(is_near(column<1>(make_native_matrix<double, 3, 3>(
    1, 0, 0,
    0, 2, 0,
    0, 0, 3)), make_native_matrix(0., 2, 0)));
  EXPECT_TRUE(is_near(column<1>(MatrixTraits<native_matrix_t<double, 3, 3>>::zero()), make_native_matrix(0., 0, 0)));
  native_matrix_t<double, 3, 3> d1;
  d1 <<
  1, 0, 0,
  0, 2, 0,
  0, 0, 3;
  const auto d_reset = d1;
  apply_columnwise(d1, [](auto& col){ col += col.Constant(1); });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& col){ return col + col.Constant(1); }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_columnwise(MatrixTraits<native_matrix_t<double, 3, 3>>::zero(),
    [](const auto& col){ return col + col.Constant(1); }), native_matrix_t<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_columnwise(d1, [](auto& col, std::size_t i){ col += col.Constant(i); });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));
  EXPECT_TRUE(is_near(apply_columnwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      0, 3, 2,
      0, 1, 5)));
  EXPECT_TRUE(is_near(apply_columnwise<3>([] { return make_native_matrix(1., 2, 3); }),
    make_native_matrix<double, 3, 3>(
      1, 1, 1,
      2, 2, 2,
      3, 3, 3)));
  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i) { return make_native_matrix(1. + i, 2 + i, 3 + i); }),
    make_native_matrix<double, 3, 3>(
      1, 2, 3,
      2, 3, 4,
      3, 4, 5)));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x){ x += 1; });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& x){ return x + 1; }),
    make_native_matrix<double, 3, 3>(
      2, 1, 1,
      1, 3, 1,
      1, 1, 4)));
  EXPECT_TRUE(is_near(apply_coefficientwise(MatrixTraits<native_matrix_t<double, 3, 3>>::zero(),
    [](const auto& x){ return x + 1; }), native_matrix_t<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x, std::size_t i, std::size_t j){ x += i + j; });
  EXPECT_TRUE(is_near(d1,
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
  EXPECT_TRUE(is_near(apply_coefficientwise(make_native_matrix<double, 3, 3>(
      1, 0, 0,
      0, 2, 0,
      0, 0, 3), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    make_native_matrix<double, 3, 3>(
      1, 1, 2,
      1, 4, 3,
      2, 3, 7)));
}
