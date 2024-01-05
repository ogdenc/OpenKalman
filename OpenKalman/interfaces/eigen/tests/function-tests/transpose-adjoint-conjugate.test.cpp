/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "interfaces/eigen/tests/eigen.gtest.hpp"

using namespace OpenKalman;
using namespace OpenKalman::Eigen3;
using namespace OpenKalman::test;


TEST(eigen3, transpose_adjoint_conjugate_matrix)
{
  auto m23 = make_dense_object_from<M23>(1, 2, 3, 4, 5, 6);
  auto m2x_3 = M2x {m23};
  auto mx3_2 = Mx3 {m23};
  auto mxx_23 = Mxx {m23};

  auto m32 = make_dense_object_from<M32>(1, 4, 2, 5, 3, 6);

  EXPECT_TRUE(is_near(transpose(m23), m32));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m23)), m32));
  EXPECT_TRUE(is_near(transpose(m23.array()), m32));
  EXPECT_TRUE(is_near(transpose(m2x_3), m32));
  EXPECT_TRUE(is_near(transpose(mx3_2), m32));
  EXPECT_TRUE(is_near(transpose(mxx_23), m32));
  EXPECT_TRUE(is_near(transpose(M23{m23}), m32));
  EXPECT_TRUE(is_near(transpose(M2x{m23}), m32));
  EXPECT_TRUE(is_near(transpose(Mx3{m23}), m32));
  EXPECT_TRUE(is_near(transpose(Mxx{m23}), m32));
  EXPECT_TRUE(is_near(transpose(Eigen::Array<double, 2, 3>{m23}), m32));

  EXPECT_TRUE(is_near(adjoint(m23), m32));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m23)), m32));
  EXPECT_TRUE(is_near(adjoint(m23.array()), m32));
  EXPECT_TRUE(is_near(adjoint(mx3_2), m32));
  EXPECT_TRUE(is_near(adjoint(m2x_3), m32));
  EXPECT_TRUE(is_near(adjoint(mxx_23), m32));
  EXPECT_TRUE(is_near(adjoint(M23{m23}), m32));
  EXPECT_TRUE(is_near(adjoint(M2x{m23}), m32));
  EXPECT_TRUE(is_near(adjoint(Mx3{m23}), m32));
  EXPECT_TRUE(is_near(adjoint(Mxx{m23}), m32));
  EXPECT_TRUE(is_near(adjoint(Eigen::Array<double, 2, 3>{m23}), m32));

  EXPECT_TRUE(is_near(conjugate(m23), m23));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m23)), m23));
  EXPECT_TRUE(is_near(conjugate(m23.array()), m23));
  EXPECT_TRUE(is_near(conjugate(mx3_2), m23));
  EXPECT_TRUE(is_near(conjugate(m2x_3), m23));
  EXPECT_TRUE(is_near(conjugate(mxx_23), m23));
  EXPECT_TRUE(is_near(conjugate(M23{m23}), m23));
  EXPECT_TRUE(is_near(conjugate(M2x{m23}), m23));
  EXPECT_TRUE(is_near(conjugate(Mx3{m23}), m23));
  EXPECT_TRUE(is_near(conjugate(Mxx{m23}), m23));
  EXPECT_TRUE(is_near(conjugate(Eigen::Array<double, 2, 3>{m23}), m23));

  auto cm23 = make_dense_object_from<CM23>(cdouble {1,6}, cdouble {2,5}, cdouble {3,4}, cdouble {4,3}, cdouble {5,2}, cdouble {6,1});
  auto cm32 = make_dense_object_from<CM32>(cdouble {1,6}, cdouble {4,3}, cdouble {2,5}, cdouble {5,2}, cdouble {3,4}, cdouble {6,1});
  auto cm32conj = make_dense_object_from<CM32>(cdouble {1,-6}, cdouble {4,-3}, cdouble {2,-5}, cdouble {5,-2}, cdouble {3,-4}, cdouble {6,-1});

  EXPECT_TRUE(is_near(transpose(cm23), cm32));
  EXPECT_TRUE(is_near(transpose(CM23{cm23}), cm32));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm23)), cm32));
  EXPECT_TRUE(is_near(transpose(cm23.array()), cm32));
  EXPECT_TRUE(is_near(adjoint(cm23), cm32conj));
  EXPECT_TRUE(is_near(adjoint(CM23{cm23}), cm32conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm23)), cm32conj));
  EXPECT_TRUE(is_near(adjoint(cm23.array()), cm32conj));
  EXPECT_TRUE(is_near(conjugate(cm32), cm32conj));
  EXPECT_TRUE(is_near(conjugate(CM32{cm32}), cm32conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm32)), cm32conj));
  EXPECT_TRUE(is_near(conjugate(cm32.array()), cm32conj));
}


TEST(eigen3, transpose_adjoint_conjugate_self_adjoint)
{
  auto m22_93310 = make_dense_object_from<M22>(9, 3, 3, 10);
  auto m20_93310 = M2x {m22_93310};
  auto m02_93310 = Mx2 {m22_93310};
  auto m00_93310 = Mxx {m22_93310};

  EXPECT_TRUE(is_near(transpose(m22_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Lower>())), m22_93310));
  EXPECT_TRUE(is_near(transpose(m20_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m02_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(transpose(m00_93310.template selfadjointView<Eigen::Upper>()), m22_93310));

  EXPECT_TRUE(is_near(adjoint(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Upper>())), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(adjoint(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  EXPECT_TRUE(is_near(conjugate(m22_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m22_93310.template selfadjointView<Eigen::Upper>())), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m20_93310.template selfadjointView<Eigen::Lower>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m02_93310.template selfadjointView<Eigen::Upper>()), m22_93310));
  EXPECT_TRUE(is_near(conjugate(m00_93310.template selfadjointView<Eigen::Lower>()), m22_93310));

  auto cm22_93310 = make_dense_object_from<CM22>(9, cdouble(3,1), cdouble(3,-1), 10);
  auto cm22_93310_trans = make_dense_object_from<CM22>(9, cdouble(3,-1), cdouble(3,1), 10);

  EXPECT_TRUE(is_near(transpose(cm22_93310.template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(std::as_const(cm22_93310).template selfadjointView<Eigen::Upper>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Upper>())), cm22_93310_trans));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(std::as_const(cm22_93310).template selfadjointView<Eigen::Upper>())), cm22_93310_trans));
  EXPECT_TRUE(is_near(adjoint(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Lower>())), cm22_93310));
  EXPECT_TRUE(is_near(conjugate(cm22_93310.template selfadjointView<Eigen::Lower>()), cm22_93310_trans));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_93310.template selfadjointView<Eigen::Lower>())), cm22_93310_trans));
}



TEST(eigen3, transpose_adjoint_conjugate_triangular)
{
  auto m22_3103 = make_dense_object_from<M22>(3, 1, 0, 3);
  auto m20_3103 = M2x {m22_3103};
  auto m02_3103 = Mx2 {m22_3103};
  auto m00_3103 = Mxx {m22_3103};

  auto m22_3013 = make_dense_object_from<M22>(3, 0, 1, 3);
  auto m20_3013 = M2x {m22_3013};
  auto m02_3013 = Mx2 {m22_3013};
  auto m00_3013 = Mxx {m22_3013};

  EXPECT_TRUE(is_near(transpose(m22_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(m22_3013.template triangularView<Eigen::Lower>())), m22_3103));
  EXPECT_TRUE(is_near(transpose(m20_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(transpose(m02_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(transpose(m00_3103.template triangularView<Eigen::Upper>()), m22_3013));

  EXPECT_TRUE(is_near(adjoint(m22_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(m22_3103.template triangularView<Eigen::Upper>())), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m20_3013.template triangularView<Eigen::Lower>()), m22_3103));
  EXPECT_TRUE(is_near(adjoint(m02_3103.template triangularView<Eigen::Upper>()), m22_3013));
  EXPECT_TRUE(is_near(adjoint(m00_3013.template triangularView<Eigen::Lower>()), m22_3103));

  EXPECT_TRUE(is_near(conjugate(m22_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(m22_3103.template triangularView<Eigen::Upper>())), m22_3103));
  EXPECT_TRUE(is_near(conjugate(m20_3013.template triangularView<Eigen::Lower>()), m22_3013));
  EXPECT_TRUE(is_near(conjugate(m02_3103.template triangularView<Eigen::Upper>()), m22_3103));
  EXPECT_TRUE(is_near(conjugate(m00_3013.template triangularView<Eigen::Lower>()), m22_3013));

  auto cm22_3013 = make_dense_object_from<CM22>(cdouble(3,0.3), 0, cdouble(1,0.1), 3);
  auto cm22_3103 = make_dense_object_from<CM22>(cdouble(3,0.3), cdouble(1,0.1), 0, 3);
  auto cm22_3013_conj = make_dense_object_from<CM22>(cdouble(3,-0.3), 0, cdouble(1,-0.1), 3);
  auto cm22_3103_conj = make_dense_object_from<CM22>(cdouble(3,-0.3), cdouble(1,-0.1), 0, 3);

  EXPECT_TRUE(is_near(transpose(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013));
  EXPECT_TRUE(is_near(transpose(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103));
  EXPECT_TRUE(is_near(transpose(std::as_const(cm22_3013).template triangularView<Eigen::Lower>()), cm22_3103));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(cm22_3103.template triangularView<Eigen::Upper>())), cm22_3013));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(std::as_const(cm22_3103).template triangularView<Eigen::Upper>())), cm22_3013));

  EXPECT_TRUE(is_near(adjoint(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(adjoint(std::as_const(cm22_3103).template triangularView<Eigen::Upper>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(adjoint(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3103_conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(cm22_3013.template triangularView<Eigen::Lower>())), cm22_3103_conj));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(std::as_const(cm22_3013).template triangularView<Eigen::Lower>())), cm22_3103_conj));

  EXPECT_TRUE(is_near(conjugate(cm22_3013.template triangularView<Eigen::Lower>()), cm22_3013_conj));
  EXPECT_TRUE(is_near(conjugate(cm22_3103.template triangularView<Eigen::Upper>()), cm22_3103_conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_3013.template triangularView<Eigen::Lower>())), cm22_3013_conj));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(cm22_3103.template triangularView<Eigen::Upper>())), cm22_3103_conj));
}


TEST(eigen3, transpose_adjoint_conjugate_diagonal)
{
  auto m21 = M21 {1, 4};
  auto m2x_1 = M2x {m21};
  auto mx1_2 = Mx1 {m21};
  auto mxx_21 = Mxx {m21};

  auto m22_1004 = make_dense_object_from<M22>(1, 0, 0, 4);

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21})), m22_1004));

  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalMatrix<double, 2> {m21}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(Eigen::DiagonalMatrix<double, 2> {m21})), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalMatrix<double, Eigen::Dynamic> {m21}), m22_1004));

  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(transpose(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(adjoint(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));

  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<M21> {m21}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen3::make_eigen_wrapper(Eigen::DiagonalWrapper<M21> {m21})), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<M2x> {m2x_1}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<Mx1> {mx1_2}), m22_1004));
  EXPECT_TRUE(is_near(conjugate(Eigen::DiagonalWrapper<Mxx> {mxx_21}), m22_1004));
}

