/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGENCHOLESKY_H
#define OPENKALMAN_EIGENCHOLESKY_H

namespace OpenKalman
{
#ifdef __cpp_concepts
  template<native_Eigen_type Arg> requires is_diagonal_v<Arg>
#else
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg> and is_diagonal_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg)
  {
    if constexpr(is_identity_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(is_1by1_v<Arg>)
    {
      return strict(std::forward<Arg>(arg).array().square().matrix());
    }
    else
    {
      return EigenDiagonal(std::forward<Arg>(arg).diagonal().array().square().matrix());;
    }
  }


#ifdef __cpp_concepts
  template<native_Eigen_type Arg> requires is_diagonal_v<Arg>
#else
  template<typename Arg, std::enable_if_t<is_native_Eigen_type_v<Arg> and is_diagonal_v<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg)
  {
    if constexpr(is_identity_v<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(is_1by1_v<Arg>)
    {
      return strict(std::forward<Arg>(arg).cwiseSqrt());
    }
    else
    {
      return EigenDiagonal(std::forward<Arg>(arg).diagonal().cwiseSqrt());;
    }
  }


  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_EigenZero_v<Arg>, int> = 0>
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg);
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    auto b = base_matrix(std::forward<Arg>(arg)).array().square().matrix();
    return EigenDiagonal<decltype(b)>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenDiagonal_v<Arg>, int> = 0>
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    auto b = base_matrix(std::forward<Arg>(arg)).cwiseSqrt();
    return EigenDiagonal<decltype(b)>(std::move(b));
  }


  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    using BaseMatrix = typename MatrixTraits<Arg>::BaseMatrix;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    static_assert(is_diagonal_v<BaseMatrix> or MatrixTraits<Arg>::storage_type == TriangleType::diagonal);

    if constexpr(is_identity_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::identity();
    }
    else if constexpr (is_zero_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::zero();
    }
    else if constexpr(is_EigenDiagonal_v<BaseMatrix>)
    {
      return Cholesky_square(base_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (is_1by1_v<BaseMatrix>)
    {
      return base_matrix(std::forward<Arg>(arg)).array().square().matrix();
    }
    else // if constexpr (not is_diagonal_v<BaseMatrix> and MatrixTraits<Arg>::storage_type == TriangleType::diagonal)
    {
      constexpr auto dimension = MatrixTraits<Arg>::dimension;
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      M b = base_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array() * b.array()).matrix();
      return EigenDiagonal(std::move(ret));
    }
  }


  template<TriangleType triangle_type, typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto
  Cholesky_factor(Arg&& arg)
  {
    using BaseMatrix = std::decay_t<typename MatrixTraits<Arg>::BaseMatrix>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dimension = MatrixTraits<Arg>::dimension;
    using M = Eigen::Matrix<Scalar, dimension, dimension>;

    if constexpr(is_identity_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::identity();
    }
    else if constexpr (is_zero_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::zero();
    }
    else if constexpr (is_EigenDiagonal_v<BaseMatrix>)
    {
      return Cholesky_factor(base_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (is_1by1_v<BaseMatrix>)
    {
      return strict(base_matrix(std::forward<Arg>(arg)).cwiseSqrt());
    }
    else if constexpr (MatrixTraits<Arg>::storage_type == TriangleType::diagonal)
    {
      using M1 = Eigen::Matrix<Scalar, dimension, 1>;
      M1 b = base_matrix(std::forward<Arg>(arg)).diagonal();
      M1 ret = (b.array().sqrt()).matrix();
      return EigenDiagonal(std::move(ret));
    }
    else if constexpr (std::is_same_v<const BaseMatrix, const typename Eigen::MatrixBase<BaseMatrix>::ConstantReturnType>)
    {
      // If base matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.
      auto s = base_matrix(std::forward<Arg>(arg)).functor()();
      if (s < Scalar(0))
      {
        // Cholesky factors are complex, so throw an exception.
        throw (std::runtime_error("Cholesky_factor of constant EigenSelfAdjointMatrix: covariance is indefinite"));
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
      return EigenTriangularMatrix<decltype(b), triangle_type>(std::move(b));
    }
    else
    {
      // For the general case, perform an LLT Cholesky decomposition.
      M b;
      auto LL_x = arg.base_view().llt();
      if (LL_x.info() == Eigen::Success)
      {
        if constexpr(triangle_type == MatrixTraits<Arg>::storage_type)
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
        auto LDL_x = base_matrix(std::forward<Arg>(arg)).ldlt();
        if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
        {
          if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
          {
            using BM = typename MatrixTraits<Arg>::BaseMatrix;
            if constexpr(triangle_type == TriangleType::lower)
              b.template triangularView<Eigen::Lower>() = MatrixTraits<BM>::zero();
           else
              b.template triangularView<Eigen::Upper>() = MatrixTraits<BM>::zero();
          }
          else // Covariance is indefinite, so throw an exception.
          {
            throw (std::runtime_error("Cholesky_factor of EigenSelfAdjointMatrix: covariance is indefinite"));
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
      return EigenTriangularMatrix<decltype(b), triangle_type>(std::move(b));
    }

  }


  template<typename Arg, std::enable_if_t<is_EigenSelfAdjointMatrix_v<Arg>, int> = 0>
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    return Cholesky_factor<MatrixTraits<Arg>::storage_type>(std::forward<Arg>(arg));
  }


  template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto
  Cholesky_square(Arg&& arg) noexcept
  {
    using BaseMatrix = std::decay_t<typename MatrixTraits<Arg>::BaseMatrix>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dimension = MatrixTraits<Arg>::dimension;
    constexpr auto triangle_type = MatrixTraits<Arg>::triangle_type;

    if constexpr(is_identity_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::identity();
    }
    else if constexpr (is_zero_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::zero();
    }
    else if constexpr (is_EigenDiagonal_v<BaseMatrix>)
    {
      return Cholesky_square(base_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (is_1by1_v<BaseMatrix>)
    {
      return strict(base_matrix(std::forward<Arg>(arg)).array().square().matrix());
    }
    else if constexpr (triangle_type == TriangleType::diagonal)
    {
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      auto b = base_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array() * b.array()).matrix();
      return EigenDiagonal(std::move(ret));
    }
    else if constexpr (triangle_type == TriangleType::upper)
    {
      using M = Eigen::Matrix<Scalar, dimension, dimension>;
      decltype(auto) b = std::forward<Arg>(arg).base_view();
      M bx = b;
      M ret = b.adjoint() * bx;
      return EigenSelfAdjointMatrix<M, triangle_type>(std::move(ret));
    }
    else // if constexpr (MatrixTraits<Arg>::triangle_type == TriangleType::lower)
    {
      using M = Eigen::Matrix<Scalar, dimension, dimension>;
      decltype(auto) b = std::forward<Arg>(arg).base_view();
      M bT = b.adjoint();
      M ret = b * bT;
      return EigenSelfAdjointMatrix<M, triangle_type>(std::move(ret));
    }

  }


  template<typename Arg, std::enable_if_t<is_EigenTriangularMatrix_v<Arg>, int> = 0>
  inline auto
  Cholesky_factor(Arg&& arg) noexcept
  {
    using BaseMatrix = std::decay_t<typename MatrixTraits<Arg>::BaseMatrix>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    static_assert(is_diagonal_v<BaseMatrix> or MatrixTraits<Arg>::triangle_type == TriangleType::diagonal);

    if constexpr(is_identity_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::identity();
    }
    else if constexpr (is_zero_v<BaseMatrix>)
    {
      return MatrixTraits<BaseMatrix>::zero();
    }
    else if constexpr (is_EigenDiagonal_v<BaseMatrix>)
    {
      return Cholesky_factor(base_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (is_1by1_v<BaseMatrix>)
    {
      return base_matrix(std::forward<Arg>(arg)).cwiseSqrt();
    }
    else // if constexpr (not is_diagonal_v<BaseMatrix> and triangle_type == TriangleType::diagonal)
    {
      constexpr auto dimension = MatrixTraits<Arg>::dimension;
      using M = Eigen::Matrix<Scalar, dimension, 1>;
      M b = base_matrix(std::forward<Arg>(arg)).diagonal();
      M ret = (b.array().sqrt()).matrix();
      return EigenDiagonal(std::move(ret));
    }

  }



}


#endif //OPENKALMAN_EIGENCHOLESKY_H
