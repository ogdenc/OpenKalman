/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_EIGEN3_CHOLESKY_H
#define OPENKALMAN_EIGEN3_CHOLESKY_H

namespace OpenKalman::Eigen3
{
  namespace detail
  {
    template<typename Arg>
    inline auto extracted_diagonal_square(Arg&& arg)
    {
      using M = Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::rows, 1>;
      M b {std::forward<Arg>(arg).diagonal()};
      M ret {(b.array() * b.array()).matrix()};
      return DiagonalMatrix<M> {std::move(ret)};
    }

    template<typename Arg>
    inline auto extracted_diagonal_sqrt(Arg&& arg)
    {
      using M = Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, MatrixTraits<Arg>::rows, 1>;
      M b {std::forward<Arg>(arg).diagonal()};
      M ret {(b.array().sqrt()).matrix()};
      return DiagonalMatrix<M> {std::move(ret)};
    }

  } // namespace detail


#ifdef __cpp_concepts
  template<eigen_native Arg> requires diagonal_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_native<Arg> and diagonal_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(one_by_one_matrix<Arg>)
    {
      return make_self_contained(std::forward<Arg>(arg).array().square().matrix());
    }
    else
    {
      auto d = make_self_contained(std::forward<Arg>(arg).diagonal().array().square().matrix());
      using D = std::decay_t<decltype(d)>;
      return DiagonalMatrix<D> {std::move(d)};
    }
  }


#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_native Arg> requires diagonal_matrix<Arg>
#else
  template<TriangleType = TriangleType::diagonal,
    typename Arg, std::enable_if_t<eigen_native<Arg> and diagonal_matrix<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<Arg> or zero_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr(one_by_one_matrix<Arg>)
    {
      return make_self_contained(std::forward<Arg>(arg).cwiseSqrt());
    }
    else
    {
      auto d = make_self_contained(std::forward<Arg>(arg).diagonal().cwiseSqrt());
      using D = std::decay_t<decltype(d)>;
      return DiagonalMatrix<D> {std::move(d)};
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
  template<TriangleType = TriangleType::diagonal, eigen_zero_expr Arg>
#else
  template<TriangleType = TriangleType::diagonal, typename Arg, std::enable_if_t<eigen_zero_expr<Arg>, int> = 0>
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
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<nested_matrix_t<Arg>> or zero_matrix<nested_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      auto d = make_self_contained(nested_matrix(std::forward<Arg>(arg)).array().square().matrix());
      using D = std::decay_t<decltype(d)>;
      return DiagonalMatrix<D> {std::move(d)};
    }
  }


#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_diagonal_expr Arg>
#else
  template<TriangleType = TriangleType::diagonal, typename Arg, std::enable_if_t<eigen_diagonal_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<nested_matrix_t<Arg>> or zero_matrix<nested_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      auto d = make_self_contained(nested_matrix(std::forward<Arg>(arg)).cwiseSqrt());
      using D = std::decay_t<decltype(d)>;
      return DiagonalMatrix<D> {std::move(d)};
    }
  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg> requires
    diagonal_matrix<nested_matrix_t<Arg>> or (MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal)
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg> and
    (diagonal_matrix<nested_matrix_t<Arg>> or (MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal)), int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<nested_matrix_t<Arg>> or zero_matrix<nested_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<Arg>>)
    {
      return Cholesky_square(nested_matrix(std::forward<Arg>(arg)));
    }
    else // if constexpr MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal
    {
      return detail::extracted_diagonal_square(nested_matrix(std::forward<Arg>(arg)));
    }
  }


#ifdef __cpp_concepts
  template<TriangleType triangle_type, eigen_self_adjoint_expr Arg>
#else
  template<TriangleType triangle_type, typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg)
  {
    using NestedMatrix = std::decay_t<nested_matrix_t<Arg>>;
    using Scalar = typename MatrixTraits<Arg>::Scalar;
    constexpr auto dimension = MatrixTraits<Arg>::rows;
    using M = Eigen::Matrix<Scalar, dimension, dimension>;

    if constexpr(identity_matrix<NestedMatrix> or zero_matrix<NestedMatrix>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<NestedMatrix>)
    {
      return Cholesky_factor<triangle_type>(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (MatrixTraits<Arg>::storage_triangle == TriangleType::diagonal)
    {
      return detail::extracted_diagonal_sqrt(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (std::is_same_v<
      const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
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
      return TriangularMatrix<M, triangle_type> {std::move(b)};
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
      return TriangularMatrix<M, triangle_type> {std::move(b)};
    }

  }


#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_self_adjoint_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    return Cholesky_factor<MatrixTraits<Arg>::storage_triangle>(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<eigen_triangular_expr Arg>
#else
  template<typename Arg, std::enable_if_t<eigen_triangular_expr<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(Arg&& arg) noexcept
  {
    constexpr auto triangle_type = MatrixTraits<Arg>::triangle_type;
    constexpr auto dim = MatrixTraits<Arg>::rows;
    using M = Eigen::Matrix<typename MatrixTraits<Arg>::Scalar, dim, dim>;

    if constexpr(identity_matrix<nested_matrix_t<Arg>> or zero_matrix<nested_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<Arg>>)
    {
      return Cholesky_square(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (triangle_type == TriangleType::diagonal)
    {
      return detail::extracted_diagonal_square(nested_matrix(std::forward<Arg>(arg)));
    }
    else if constexpr (triangle_type == TriangleType::upper)
    {
      auto v = std::forward<Arg>(arg).nested_view();
      M ret;
      ret.template triangularView<Eigen::Upper>() = v.adjoint() * M {v};
      return SelfAdjointMatrix<M, triangle_type> {std::move(ret)};
    }
    else // if constexpr (triangle_type == TriangleType::lower)
    {
      auto v = std::forward<Arg>(arg).nested_view();
      M ret;
      ret.template triangularView<Eigen::Lower>() = v * M {v.adjoint()};
      return SelfAdjointMatrix<M, triangle_type> {std::move(ret)};
    }

  }


#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_triangular_expr Arg> requires
    diagonal_matrix<nested_matrix_t<Arg>> or (MatrixTraits<Arg>::triangle_type == TriangleType::diagonal)
#else
  template<TriangleType = TriangleType::diagonal, typename Arg, std::enable_if_t<eigen_triangular_expr<Arg> and
    (diagonal_matrix<nested_matrix_t<Arg>> or (MatrixTraits<Arg>::triangle_type == TriangleType::diagonal)), int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(Arg&& arg) noexcept
  {
    if constexpr(identity_matrix<nested_matrix_t<Arg>> or zero_matrix<nested_matrix_t<Arg>>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<Arg>>)
    {
      return Cholesky_factor(nested_matrix(std::forward<Arg>(arg)));
    }
    else // if constexpr (MatrixTraits<Arg>::triangle_type == TriangleType::diagonal)
    {
      return detail::extracted_diagonal_sqrt(nested_matrix(std::forward<Arg>(arg)));
    }

  }

}


#endif //OPENKALMAN_EIGEN3_CHOLESKY_H
