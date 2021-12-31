/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "eigen3.gtest.hpp"
#include <complex>

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;

namespace
{
  using M11 = eigen_matrix_t<double, 1, 1>;
  using M12 = eigen_matrix_t<double, 1, 2>;
  using M13 = eigen_matrix_t<double, 1, 3>;
  using M21 = eigen_matrix_t<double, 2, 1>;
  using M22 = eigen_matrix_t<double, 2, 2>;
  using M23 = eigen_matrix_t<double, 2, 3>;
  using M31 = eigen_matrix_t<double, 3, 1>;
  using M32 = eigen_matrix_t<double, 3, 2>;
  using M33 = eigen_matrix_t<double, 3, 3>;
  using M34 = eigen_matrix_t<double, 3, 4>;
  using M43 = eigen_matrix_t<double, 4, 3>;
  using M44 = eigen_matrix_t<double, 4, 4>;
  using M55 = eigen_matrix_t<double, 5, 5>;

  using M00 = eigen_matrix_t<double, dynamic_extent, dynamic_extent>;
  using M10 = eigen_matrix_t<double, 1, dynamic_extent>;
  using M01 = eigen_matrix_t<double, dynamic_extent, 1>;
  using M20 = eigen_matrix_t<double, 2, dynamic_extent>;
  using M02 = eigen_matrix_t<double, dynamic_extent, 2>;
  using M30 = eigen_matrix_t<double, 3, dynamic_extent>;
  using M03 = eigen_matrix_t<double, dynamic_extent, 3>;
  using M04 = eigen_matrix_t<double, dynamic_extent, 4>;
  using M50 = eigen_matrix_t<double, 5, dynamic_extent>;
  using M05 = eigen_matrix_t<double, dynamic_extent, 5>;

  using cdouble = std::complex<double>;

  using CM21 = eigen_matrix_t<cdouble, 2, 1>;
  using CM22 = eigen_matrix_t<cdouble, 2, 2>;
  using CM23 = eigen_matrix_t<cdouble, 2, 3>;
  using CM32 = eigen_matrix_t<cdouble, 3, 2>;
  using CM34 = eigen_matrix_t<cdouble, 3, 4>;
  using CM43 = eigen_matrix_t<cdouble, 4, 3>;
}


TEST(eigen3, randomize)
{
  using N = std::normal_distribution<double>;

  M22 m22, m22_true;
  M23 m23, m23_true;
  M32 m32, m32_true;
  M20 m20_2 {2, 2};
  M20 m20_3 {2, 3};
  M02 m02_2 {2, 2};
  M02 m02_3 {2, 2};
  M30 m30 {3, 2};
  M03 m03 {2, 3};
  M00 m00 {2, 2};

  // Test just using the parameters, rather than a constructed distribution.
  m22 = randomize<M22>(N {0.0, 0.7});
  m20_2 = randomize<M20>(2, 2, N {0.0, 1.0});
  m20_3 = randomize<M20>(2, 3, N {0.0, 0.7});
  m02_2 = randomize<M02>(2, 2, N {0.0, 1.0});
  m02_3 = randomize<M02>(3, 2, N {0.0, 0.7});
  m00 = randomize<M00>(2, 2, N {0.0, 1.0});

  // Single distribution for the entire matrix.
  m22 = M22::Zero();
  m20_2 = M20::Zero(2, 2);
  m02_2 = M02::Zero(2, 2);
  m00 = M00::Zero(2, 2);
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3})) / (i + 1);
    m20_2 = (m20_2 * i + randomize<M20>(2, 2, N {1.0, 0.3})) / (i + 1);
    m02_2 = (m02_2 * i + randomize<M02>(2, 2, N {1.0, 0.3})) / (i + 1);
    m00 = (m00 * i + randomize<M00>(2, 2, N {1.0, 0.3})) / (i + 1);
  }
  m22_true = M22::Constant(1);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m20_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m20_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m02_2, m22_true, 0.1));
  EXPECT_FALSE(is_near(m02_2, m22_true, 1e-8));
  EXPECT_TRUE(is_near(m00, m22_true, 0.1));
  EXPECT_FALSE(is_near(m00, m22_true, 1e-8));

  // A distribution for each element.
  m22 = M22::Zero();
  for (int i=0; i<100; i++)
  {
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})) / (i + 1);
  }
  m22_true = MatrixTraits<M22>::make(1, 2, 3, 4);
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each row.
  m32 = M32::Zero();
  m22 = M22::Zero();
  for (int i=0; i<100; i++)
  {
    m32 = (m32 * i + randomize<M32>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
    m22 = (m22 * i + randomize<M22>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
  }
  m32_true = MatrixTraits<M32>::make(1, 1, 2, 2, 3, 3);
  m22_true = MatrixTraits<M22>::make(1, 1, 2, 2);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m22, m22_true, 0.1));
  EXPECT_FALSE(is_near(m22, m22_true, 1e-8));

  // One distribution for each column.
  m32 = M32::Zero();
  m23 = M23::Zero();
  for (int i=0; i<100; i++)
  {
    m32 = (m32 * i + randomize<M32>(N {1.0, 0.3}, N {2.0, 0.3})) / (i + 1);
    m23 = (m23 * i + randomize<M23>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})) / (i + 1);
  }
  m32_true = MatrixTraits<M32>::make(1, 2, 1, 2, 1, 2);
  m23_true = MatrixTraits<M23>::make(1, 2, 3, 1, 2, 3);
  EXPECT_TRUE(is_near(m32, m32_true, 0.1));
  EXPECT_FALSE(is_near(m32, m32_true, 1e-8));
  EXPECT_TRUE(is_near(m23, m23_true, 0.1));
  EXPECT_FALSE(is_near(m23, m23_true, 1e-8));
}

