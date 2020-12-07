/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_CHOLESKY_H
#define OPENKALMAN_EIGEN3_CHOLESKY_H

namespace OpenKalman::Eigen3
{
#ifdef __cpp_concepts
  template<eigen_native Arg> requires diagonal_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and diagonal_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg)
  {
    if constexpr(identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(one_by_one_matrix<Arg>)
    {
      return make_self_contained(std::forward<Arg>(arg).array().square().matrix());
    }
    else
    {
      return DiagonalMatrix(std::forward<Arg>(arg).diagonal().array().square().matrix());;
    }
  }


#ifdef __cpp_concepts
  template<eigen_native Arg> requires diagonal_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and diagonal_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg)
  {
    if constexpr(identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(one_by_one_matrix<Arg>)
    {
      return make_self_contained(std::forward<Arg>(arg).cwiseSqrt());
    }
    else
    {
      return DiagonalMatrix(std::forward<Arg>(arg).diagonal().cwiseSqrt());;
    }
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_zero_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    auto b = nested_matrix(std::forward<Arg>(arg)).array().square().matrix();
    return DiagonalMatrix<decltype(b)>(std::move(b));
  }


#ifdef __cpp_concepts
  template<eigen_diagonal_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    auto b = nested_matrix(std::forward<Arg>(arg)).cwiseSqrt();
    return DiagonalMatrix<decltype(b)>(std::move(b));
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    using NestedMatrix = nested_matrix_t<Arg>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    static_assert(diagonal_matrix<NestedMatrix> or MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal);

    if constexpr(identity_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::identity();
    }
    else if constexpr (zero_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::zero();
    }
    else if constexpr(eigen_diagonal_expr<NestedMatrix>)
    {
      return Cholesky_square(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (one_by_one_matrix<NestedMatrix>)
    {
      return nested_matrix(std::forward<Arg>(arg)).array().square().matrix();
    }
    else // if constexpr (not diagonal_matrix<NestedMatrix> and MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal)
    {
      constexpr auto dimension = MatrixTraits<Arg>::dimension;
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      M b = nested_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array() * b.array()).matrix();
      return DiagonalMatrix(std::move(ret));
    }
  }


#ifdef __cpp_concepts
  template<TriangleType triangle_type, eigen_self_adjoint_expr Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_factor(Arg&& arg)
  {
    using NestedMatrix = std::decay_t<nested_matrix_t<Arg>>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dimension = MatrixTraits<Arg>::dimension;
    using M = Eigen::Matrix<Scalar, dimension, dimension>;

    if constexpr(identity_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::identity();
    }
    else if constexpr (zero_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::zero();
    }
    else if constexpr (eigen_diagonal_expr<NestedMatrix>)
    {
      return Cholesky_factor(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (one_by_one_matrix<NestedMatrix>)
    {
      return make_self_contained(nested_matrix(std::forward<Arg>(arg)).cwiseSqrt());
    }
    else if constexpr (MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal)
    {
      using M1 = Eigen::Matrix<Scalar, dimension, 1>;
      M1 b = nested_matrix(std::forward<Arg>(arg)).diagonal();
      M1 ret = (b.array().sqrt()).matrix();
      return DiagonalMatrix(std::move(ret));
    }
    else if constexpr (std::is_same_v<const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
    {
      // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.
      auto s = nested_matrix(std::forward<Arg>(arg)).functor()();
      if (s < Scalar(0))
      {
        // Cholesky factors are complex, so throw an exception.
        throw (std::runtime_error("Cholesky_factor of constant SelfAdjointMatrix: covariance is indefinite"));
      }
      M b;
      if constexpr(triangle_type == TriangleType::lower)
      {
        using Mat = Eigen::Matrix<Scalar, 1, dimension>;
        Mat mat = Mat::Zero();
        mat(0, 0) = std::sqrt(s);
        b.template triangularView<Eigen::Lower>() = mat.template replicate<dimension, 1>();
      }
      else
      {
        using Mat = Eigen::Matrix<Scalar, dimension, 1>;
        Mat mat = Mat::Zero();
        mat(0, 0) = std::sqrt(s);
        b.template triangularView<Eigen::Upper>() = mat.template replicate<1, dimension>();
      }
      return TriangularMatrix<decltype(b), triangle_type>(std::move(b));
    }
    else
    {
      // For the general case, perform an LLT Cholesky decomposition.
      M b;
      auto LL_x = arg.nested_view().llt();
      if (LL_x.info() == Eigen::Success)
      {
        if constexpr(triangle_type == MatrixTraits<Arg>::storage_triangle)
        {
          b = std::move(LL_x.matrixLLT());
        }
        else
        {
          constexpr unsigned int uplo = triangle_type == TriangleType::upper ? Eigen::Upper : Eigen::Lower;
          b.template triangularView<uplo>() = LL_x.matrixLLT().adjoint();
        }
      }
      else [[unlikely]]
      {
        // If covariance is not positive definite, use the more robust LDLT decomposition.
        auto LDL_x = nested_matrix(std::forward<Arg>(arg)).ldlt();
        if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
        {
          if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
          {
            using BM = nested_matrix_t<Arg>;
            if constexpr(triangle_type == TriangleType::lower)
              b.template triangularView<Eigen::Lower>() = MatrixTraits<BM>::zero();
            else
              b.template triangularView<Eigen::Upper>() = MatrixTraits<BM>::zero();
          }
          else // Covariance is indefinite, so throw an exception.
          {
            throw (std::runtime_error("Cholesky_factor of SelfAdjointMatrix: covariance is indefinite"));
          }
        }
        else if constexpr(triangle_type == TriangleType::lower)
        {
          b.template triangularView<Eigen::Lower>() =
            LDL_x.matrixL().toDenseMatrix() * LDL_x.vectorD().cwiseSqrt().asDiagonal();
        }
        else
        {
          b.template triangularView<Eigen::Upper>() =
            LDL_x.vectorD().cwiseSqrt().asDiagonal() * LDL_x.matrixU().toDenseMatrix();
        }
      }
      return TriangularMatrix<decltype(b), triangle_type>(std::move(b));
    }

  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    return Cholesky_factor<MatrixTraits<Arg>::storage_triangle>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    using NestedMatrix = std::decay_t<nested_matrix_t<Arg>>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dimension = MatrixTraits<Arg>::dimension;
    constexpr auto triangle_type = MatrixTraits<Arg>::triangle_type;

    if constexpr(identity_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::identity();
    }
    else if constexpr (zero_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::zero();
    }
    else if constexpr (eigen_diagonal_expr<NestedMatrix>)
    {
      return Cholesky_square(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (one_by_one_matrix<NestedMatrix>)
    {
      return make_self_contained(nested_matrix(std::forward<Arg>(arg)).array().square().matrix());
    }
    else if constexpr (triangle_type == TriangleType::diagonal)
    {
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      auto b = nested_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array() * b.array()).matrix();
      return DiagonalMatrix(std::move(ret));
    }
    else if constexpr (triangle_type == TriangleType::upper)
    {
      using M = Eigen::Matrix<Scalar, dimension, dimension>;
      decltype(auto) b = std::forward<Arg>(arg).nested_view();
      M bx = b;
      M ret = b.adjoint() * bx;
      return SelfAdjointMatrix<M, triangle_type>(std::move(ret));
    }
    else // if constexpr (MatrixTraits<Arg>::triangle_type == TriangleType::lower)
    {
      using M = Eigen::Matrix<Scalar, dimension, dimension>;
      decltype(auto) b = std::forward<Arg>(arg).nested_view();
      M bT = b.adjoint();
      M ret = b * bT;
      return SelfAdjointMatrix<M, triangle_type>(std::move(ret));
    }

  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    using NestedMatrix = std::decay_t<nested_matrix_t<Arg>>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    static_assert(diagonal_matrix<NestedMatrix> or MatrixTraits<Arg>::triangle_type == TriangleType::diagonal);

    if constexpr(identity_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::identity();
    }
    else if constexpr (zero_matrix<NestedMatrix>)
    {
      return MatrixTraits<NestedMatrix>::zero();
    }
    else if constexpr (eigen_diagonal_expr<NestedMatrix>)
    {
      return Cholesky_factor(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (one_by_one_matrix<NestedMatrix>)
    {
      return nested_matrix(std::forward<Arg>(arg)).cwiseSqrt();
    }
    else // if constexpr (not diagonal_matrix<NestedMatrix> and triangle_type == TriangleType::diagonal)
    {
      constexpr auto dimension = MatrixTraits<Arg>::dimension;
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      M b = nested_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array().sqrt()).matrix();
      return DiagonalMatrix(std::move(ret));
    }

  }

}


#endif //OPENKALMAN_EIGEN3_CHOLESKY_H
