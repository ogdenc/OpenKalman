/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "matrix_tests.h"

using namespace OpenKalman;

using Mat2 = Eigen::Matrix<double, 2, 2>;
using Mat3 = Eigen::Matrix<double, 3, 3>;
using Axis2 = Coefficients<Axis, Axis>;

TEST_F(matrix_tests, Eigen_Matrix_class_traits)
{
  EXPECT_TRUE(is_near(MatrixTraits<Eigen::Matrix<double, 2, 3>>::zero(), (Eigen::Matrix<double, 2, 3>() << 0, 0, 0, 0, 0, 0).finished()));
  EXPECT_TRUE(is_near(MatrixTraits<Mat3>::identity(), (Mat3() << 1, 0, 0, 0, 1, 0, 0, 0, 1).finished()));
  EXPECT_NEAR(((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished())(0, 0), 1, 1e-6);
  EXPECT_NEAR(((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished())(0, 1), 2, 1e-6);
  auto d1 = (Eigen::Matrix<double, 3, 1>() << 1, 2, 3).finished();
  EXPECT_NEAR(d1(1), 2, 1e-6);
  d1(0) = 5;
  d1(2, 0) = 7;
  EXPECT_TRUE(is_near(d1, (Eigen::Matrix<double, 3, 1>() << 5, 2, 7).finished()));
}

TEST_F(matrix_tests, Eigen_Matrix_overloads)
{
  EXPECT_TRUE(is_near(transpose((Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished()),
    (Eigen::Matrix<double, 3, 2>() << 1, 4, 2, 5, 3, 6).finished()));
  EXPECT_TRUE(is_near(adjoint((Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished()),
    (Eigen::Matrix<double, 3, 2>() << 1, 4, 2, 5, 3, 6).finished()));
  EXPECT_NEAR(determinant((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished()), -2, 1e-6);
  EXPECT_NEAR(determinant((Eigen::Matrix<double, 1, 1>() << 2).finished()), 2, 1e-6);
  EXPECT_NEAR(trace((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished()), 5, 1e-6);
  EXPECT_NEAR(trace((Eigen::Matrix<double, 1, 1>() << 3).finished()), 3, 1e-6);
  EXPECT_TRUE(is_near(solve((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished(),
    (Eigen::Matrix<double, 2, 1>() << 5, 6).finished()),
    (Eigen::Matrix<double, 2, 1>() << -4, 4.5).finished()));
  EXPECT_TRUE(is_near(solve((Eigen::Matrix<double, 1, 1>() << 2).finished(),
    (Eigen::Matrix<double, 1, 1>() << 6).finished()),
    (Eigen::Matrix<double, 1, 1>() << 3).finished()));
  EXPECT_TRUE(is_near(reduce_columns((Eigen::Matrix<double, 2, 3>() << 1, 2, 3, 4, 5, 6).finished()),
    (Eigen::Matrix<double, 2, 1>() << 2, 5).finished()));
  EXPECT_TRUE(is_near(LQ_decomposition((Eigen::Matrix<double, 2, 2>() << 0.06, 0.08, 0.36, -1.640).finished()),
    (Eigen::Matrix<double, 2, 2>() << -0.1, 0, 1.096, -1.272).finished()));
  EXPECT_TRUE(is_near(QR_decomposition((Eigen::Matrix<double, 2, 2>() << 0.06, 0.36, 0.08, -1.640).finished()),
    (Eigen::Matrix<double, 2, 2>() << -0.1, 1.096, 0, -1.272).finished()));
  //
  using Mat = Eigen::Matrix<double, 3, 1>;
  Mat m;
  m = randomize<Mat>(0.0, 1.0);
  m = randomize<Mat>(0.0, 0.7);
  m = Mat::Zero();
  for (int i=0; i<100; i++)
  {
    m = (m * i + randomize<Mat>(1.0, 0.3)) / (i + 1);
  }
  auto offset = Mat::Constant(1);
  EXPECT_TRUE(is_near(m, offset, 0.1));
  EXPECT_FALSE(is_near(m, offset, 1e-8));
}

TEST_F(matrix_tests, EigenZero)
{
  EXPECT_NEAR((EigenZero<Eigen::Matrix<double, 2, 2>>()(0, 0)), 0, 1e-6);
  EXPECT_NEAR((EigenZero<Eigen::Matrix<double, 2, 2>>()(0, 1)), 0, 1e-6);
  EXPECT_TRUE(is_near(strict_matrix(EigenZero<Eigen::Matrix<double, 2, 3>>()), Eigen::Matrix<double, 2, 3>::Zero()));
  EXPECT_TRUE(is_near(strict(EigenZero<Eigen::Matrix<double, 2, 3>>()), Eigen::Matrix<double, 2, 3>::Zero()));
  EXPECT_NEAR(determinant(EigenZero<Eigen::Matrix<double, 2, 2>>()), 0, 1e-6);
  EXPECT_NEAR(trace(EigenZero<Eigen::Matrix<double, 2, 2>>()), 0, 1e-6);
  EXPECT_TRUE(is_near(reduce_columns(EigenZero<Eigen::Matrix<double, 2, 3>>()), (Eigen::Matrix<double, 2, 1>::Zero())));
  EXPECT_TRUE(is_near(column(EigenZero<Eigen::Matrix<double, 2, 3>>(), 1), (Eigen::Matrix<double, 2, 1>::Zero())));
  EXPECT_TRUE(is_near(column<1>(EigenZero<Eigen::Matrix<double, 2, 3>>()), (Eigen::Matrix<double, 2, 1>::Zero())));
}

TEST_F(matrix_tests, Matrix_blocks)
{
  EXPECT_TRUE(is_near(concatenate_vertical((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished(),
    (Eigen::Matrix<double, 1, 2>() << 5, 6).finished()), (Eigen::Matrix<double, 3, 2>() << 1, 2, 3, 4, 5, 6).finished()));
  EXPECT_TRUE(is_near(concatenate_horizontal((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished(),
    (Eigen::Matrix<double, 2, 1>() << 5, 6).finished()), (Eigen::Matrix<double, 2, 3>() << 1, 2, 5, 3, 4, 6).finished()));
  EXPECT_TRUE(is_near(concatenate_diagonal((Eigen::Matrix<double, 2, 2>() << 1, 2, 3, 4).finished(),
    (Eigen::Matrix<double, 2, 2>() << 5, 6, 7, 8).finished()), (Eigen::Matrix<double, 4, 4>() << 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 0, 7, 8).finished()));
  EXPECT_TRUE(is_near(split_vertical((Eigen::Matrix<double, 2, 2>() << 1, 0, 0, 2).finished()), std::tuple{}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>((Eigen::Matrix<double, 5, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3,
    4, 0, 0,
    0, 5, 0).finished()),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(),
               (Eigen::Matrix<double, 2, 3>() <<
                 4, 0, 0,
                 0, 5, 0).finished()}));
  EXPECT_TRUE(is_near(split_vertical<2, 2>((Eigen::Matrix<double, 5, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3,
      4, 0, 0,
      0, 5, 0).finished()),
    std::tuple{(Eigen::Matrix<double, 2, 3>() <<
      1, 0, 0,
      0, 2, 0).finished(),
               (Eigen::Matrix<double, 2, 3>() <<
                 0, 0, 3,
                 4, 0, 0).finished()}));
  EXPECT_TRUE(is_near(split_vertical<3, 2>(MatrixTraits<Eigen::Matrix<double, 5, 3>>::zero()),
    std::tuple {Eigen::Matrix<double, 3, 3>::Zero(), Eigen::Matrix<double, 2, 3>::Zero()}));
  EXPECT_TRUE(is_near(split_horizontal((Eigen::Matrix<double, 2, 2>() << 1, 0, 0, 2).finished()), std::tuple{}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>((Eigen::Matrix<double, 3, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5).finished()),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(),
               (Eigen::Matrix<double, 3, 2>() <<
                 0, 0,
                 4, 0,
                 0, 5).finished()}));
  EXPECT_TRUE(is_near(split_horizontal<2, 2>((Eigen::Matrix<double, 3, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 4, 0,
      0, 0, 3, 0, 5).finished()),
    std::tuple{(Eigen::Matrix<double, 3, 2>() <<
      1, 0,
      0, 2,
      0, 0).finished(),
               (Eigen::Matrix<double, 3, 2>() <<
                 0, 0,
                 0, 4,
                 3, 0).finished()}));
  EXPECT_TRUE(is_near(split_horizontal<3, 2>(MatrixTraits<Eigen::Matrix<double, 3, 5>>::zero()),
    std::tuple {Eigen::Matrix<double, 3, 3>::Zero(), Eigen::Matrix<double, 3, 2>::Zero()}));
  EXPECT_TRUE(is_near(split_diagonal((Eigen::Matrix<double, 2, 2>() << 1, 0, 0, 2).finished()), std::tuple{}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>((Eigen::Matrix<double, 5, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5).finished()),
    std::tuple{(Eigen::Matrix<double, 3, 3>() <<
        1, 0, 0,
        0, 2, 0,
        0, 0, 3).finished(),
               (Eigen::Matrix<double, 2, 2>() <<
                 4, 0,
                 0, 5).finished()}));
  EXPECT_TRUE(is_near(split_diagonal<2, 2>((Eigen::Matrix<double, 5, 5>() <<
      1, 0, 0, 0, 0,
      0, 2, 0, 0, 0,
      0, 0, 3, 0, 0,
      0, 0, 0, 4, 0,
      0, 0, 0, 0, 5).finished()),
    std::tuple{(Eigen::Matrix<double, 2, 2>() <<
      1, 0,
      0, 2).finished(),
               (Eigen::Matrix<double, 2, 2>() <<
                 3, 0,
                 0, 4).finished()}));
  EXPECT_TRUE(is_near(split_diagonal<3, 2>(MatrixTraits<Eigen::Matrix<double, 5, 5>>::zero()),
    std::tuple {Eigen::Matrix<double, 3, 3>::Zero(), Eigen::Matrix<double, 2, 2>::Zero()}));
  EXPECT_TRUE(is_near(column((Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished(), 2), Mean{0., 0, 3}));
  EXPECT_TRUE(is_near(column(MatrixTraits<Eigen::Matrix<double, 3, 3>>::zero(), 2), Mean{0., 0, 0}));
  EXPECT_TRUE(is_near(column<1>((Eigen::Matrix<double, 3, 3>() <<
    1, 0, 0,
    0, 2, 0,
    0, 0, 3).finished()), Mean{0., 2, 0}));
  EXPECT_TRUE(is_near(column<1>(MatrixTraits<Eigen::Matrix<double, 3, 3>>::zero()), Mean{0., 0, 0}));
  Eigen::Matrix<double, 3, 3> d1;
  d1 <<
  1, 0, 0,
  0, 2, 0,
  0, 0, 3;
  const auto d_reset = d1;
  apply_columnwise(d1, [](auto& col){ col += col.Constant(1); });
  EXPECT_TRUE(is_near(d1,
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_columnwise((Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(), [](const auto& col){ return col + col.Constant(1); }),
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_columnwise(MatrixTraits<Eigen::Matrix<double, 3, 3>>::zero(),
    [](const auto& col){ return col + col.Constant(1); }), Eigen::Matrix<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_columnwise(d1, [](auto& col, std::size_t i){ col += col.Constant(i); });
  EXPECT_TRUE(is_near(d1,
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      0, 3, 2,
      0, 1, 5).finished()));
  EXPECT_TRUE(is_near(apply_columnwise((Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(), [](const auto& col, std::size_t i){ return col + col.Constant(i); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      0, 3, 2,
      0, 1, 5).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>([] { return Mean {1., 2, 3}.base_matrix(); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 1,
      2, 2, 2,
      3, 3, 3).finished()));
  EXPECT_TRUE(is_near(apply_columnwise<3>([](std::size_t i) { return Mean {1. + i, 2 + i, 3 + i}.base_matrix(); }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 2, 3,
      2, 3, 4,
      3, 4, 5).finished()));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x){ x += 1; });
  EXPECT_TRUE(is_near(d1,
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise((Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(), [](const auto& x){ return x + 1; }),
    (Eigen::Matrix<double, 3, 3>() <<
      2, 1, 1,
      1, 3, 1,
      1, 1, 4).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise(MatrixTraits<Eigen::Matrix<double, 3, 3>>::zero(),
    [](const auto& x){ return x + 1; }), Eigen::Matrix<double, 3, 3>::Constant(1)));
  d1 = d_reset;
  apply_coefficientwise(d1, [](auto& x, std::size_t i, std::size_t j){ x += i + j; });
  EXPECT_TRUE(is_near(d1,
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      1, 4, 3,
      2, 3, 7).finished()));
  EXPECT_TRUE(is_near(apply_coefficientwise((Eigen::Matrix<double, 3, 3>() <<
      1, 0, 0,
      0, 2, 0,
      0, 0, 3).finished(), [](const auto& x, std::size_t i, std::size_t j){ return x + i + j; }),
    (Eigen::Matrix<double, 3, 3>() <<
      1, 1, 2,
      1, 4, 3,
      2, 3, 7).finished()));
}
