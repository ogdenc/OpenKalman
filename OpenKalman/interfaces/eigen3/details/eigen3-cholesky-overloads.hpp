/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions of Eigen3::Cholesky_square and Eigen3::Cholesky_factor
 */

#ifndef OPENKALMAN_EIGEN3_CHOLESKY_HPP
#define OPENKALMAN_EIGEN3_CHOLESKY_HPP


namespace OpenKalman::Eigen3
{
  /**
   * \brief Take the Cholesky square (AA<sup>T</sup>) of an Eigen::ZeroMatrix.
   * \param z A zero matrix.
   * \return zz<sup>T</sup>. (In other words, z is unchanged.)
   */
#ifdef __cpp_concepts
  template<eigen_zero_expr Z>
#else
  template<typename Z, std::enable_if_t<eigen_zero_expr<Z>, int> = 0>
#endif
  constexpr Z&&
  Cholesky_square(Z&& z) noexcept
  {
    return std::forward<Z>(z);
  }


  /**
   * \brief Take the Cholesky factor of an Eigen::ZeroMatrix.
   * \param z A zero matrix.
   * \return z, where the argument is in the form zz<sup>T</sup>. (In other words, z is unchanged.)
   */
#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_zero_expr Z>
#else
  template<TriangleType = TriangleType::diagonal, typename Z, std::enable_if_t<eigen_zero_expr<Z>, int> = 0>
#endif
  constexpr Z&&
  Cholesky_factor(Z&& z) noexcept
  {
    return std::forward<Z>(z);
  }


  /**
   * \brief Take the Cholesky square of a diagonal native Eigen matrix.
   * \param d A native Eigen diagonal matrix.
   * \return dd<sup>T</sup>
   */
#ifdef __cpp_concepts
  template<eigen_native D> requires diagonal_matrix<D>
#else
  template<typename D, std::enable_if_t<eigen_native<D> and diagonal_matrix<D>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(D&& d) noexcept
  {
    if constexpr(identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else if constexpr(one_by_one_matrix<D>)
    {
      return make_self_contained<D>(std::forward<D>(d).array().square().matrix());
    }
    else
    {
      auto n = make_self_contained<D>(std::forward<D>(d).diagonal().array().square().matrix());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky factor of a diagonal native Eigen matrix.
   * \param d A native Eigen diagonal matrix.
   * \return e, where ee<sup>T</sup> = d.
   */
#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_native D> requires diagonal_matrix<D>
#else
  template<TriangleType = TriangleType::diagonal,
    typename D, std::enable_if_t<eigen_native<D> and diagonal_matrix<D>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(D&& d) noexcept
  {
    if constexpr(identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else if constexpr(one_by_one_matrix<D>)
    {
      return make_self_contained(std::forward<D>(d).cwiseSqrt());
    }
    else
    {
      auto n = make_self_contained<D>(std::forward<D>(d).diagonal().cwiseSqrt());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky square of an Eigen::DiagonalMatrix.
   * \param d An Eigen::DiagonalMatrix.
   * \return dd<sup>T</sup>
   */
#ifdef __cpp_concepts
  template<eigen_diagonal_expr D>
#else
  template<typename D, std::enable_if_t<eigen_diagonal_expr<D>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(D&& d) noexcept
  {
    if constexpr(identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else
    {
      auto n = make_self_contained<D>(nested_matrix(std::forward<D>(d)).array().square().matrix());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky factor of an Eigen::DiagonalMatrix.
   * \param d An Eigen::DiagonalMatrix.
   * \return e, where ee<sup>T</sup> = d.
   */
#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_diagonal_expr D>
#else
  template<TriangleType = TriangleType::diagonal, typename D, std::enable_if_t<eigen_diagonal_expr<D>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(D&& d) noexcept
  {
    if constexpr(identity_matrix<D> or zero_matrix<D>)
    {
      return std::forward<D>(d);
    }
    else
    {
      auto n = make_self_contained<D>(nested_matrix(std::forward<D>(d)).cwiseSqrt());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky square of a diagonal Eigen::SelfAdjointMatrix.
   * \param a An Eigen::SelfAdjointMatrix.
   * \return aa<sup>T</sup>
   */
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr A> requires diagonal_matrix<A>
#else
  template<typename A, std::enable_if_t<eigen_self_adjoint_expr<A> and diagonal_matrix<A>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(A&& a) noexcept
  {
    if constexpr(identity_matrix<A> or zero_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<A>>)
    {
      return Cholesky_square(nested_matrix(std::forward<A>(a)));
    }
    else
    {
      static_assert(MatrixTraits<A>::storage_triangle == TriangleType::diagonal);
      auto n = make_self_contained<A>(nested_matrix(std::forward<A>(a)).diagonal().array().square().matrix());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
  }


  /**
   * \brief Take the Cholesky factor of an Eigen::SelfAdjointMatrix.
   * \tparam triangle_type The triangle type of the result. This can only be TriangleType::diagonal if a is diagonal.
   * \param a An Eigen::SelfAdjointMatrix.
   * \return A triangle t (upper or lower), depending on triangle_type), where tt<sup>T</sup> = a or t<sup>T</sup>t = a.
   */
#ifdef __cpp_concepts
  template<TriangleType triangle_type, eigen_self_adjoint_expr A> requires
    (triangle_type != TriangleType::diagonal or diagonal_matrix<A>)
#else
  template<TriangleType triangle_type, typename A, std::enable_if_t<
    eigen_self_adjoint_expr<A> and (triangle_type != TriangleType::diagonal or diagonal_matrix<A>), int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(A&& a)
  {
    using NestedMatrix = std::decay_t<nested_matrix_t<A>>;
    using Scalar = typename MatrixTraits<A>::Scalar;
    constexpr auto dimensions = MatrixTraits<A>::rows;
    using M = Eigen::Matrix<Scalar, dimensions, dimensions>;

    if constexpr(identity_matrix<A> or zero_matrix<A>)
    {
      return std::forward<A>(a);
    }
    else if constexpr (eigen_constant_expr<NestedMatrix>)
    {
      // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

      // Check that Cholesky factor elements are real:
      constexpr auto s = MatrixTraits<NestedMatrix>::constant;
      static_assert(s >= Scalar(0), "Cholesky_factor of constant SelfAdjointMatrix: covariance is indefinite");
      constexpr auto sqrt_s = OpenKalman::internal::constexpr_sqrt(s);

      if constexpr (triangle_type == TriangleType::diagonal)
      {
        static_assert(diagonal_matrix<A>);
        using CM = ConstantMatrix<Scalar, sqrt_s, dimensions, 1>;
        return DiagonalMatrix<CM> {CM {}};
      }
      else if constexpr (triangle_type == TriangleType::lower)
      {
        ConstantMatrix<Scalar, sqrt_s, dimensions, 1> col0;
        ZeroMatrix<Scalar, dimensions, dimensions - 1> othercols;
        auto m = concatenate_horizontal(col0, othercols);
        return TriangularMatrix<std::decay_t<decltype(m)>, triangle_type> {std::move(m)};
      }
      else
      {
        static_assert(triangle_type == TriangleType::upper);
        ConstantMatrix<Scalar, sqrt_s, 1, dimensions> row0;
        ZeroMatrix<Scalar, dimensions - 1, dimensions> otherrows;
        auto m = concatenate_vertical(row0, otherrows);
        return TriangularMatrix<std::decay_t<decltype(m)>, triangle_type> {std::move(m)};
      }
    }
    else if constexpr (std::is_same_v<
      const NestedMatrix, const typename Eigen::MatrixBase<NestedMatrix>::ConstantReturnType>)
    {
      // If nested matrix is a positive constant matrix, construct the Cholesky factor using a shortcut.

      auto s = nested_matrix(std::forward<A>(a)).functor()();

      if (s < Scalar(0))
      {
        // Cholesky factor elements are complex, so throw an exception.
        throw (std::runtime_error("Cholesky_factor of constant SelfAdjointMatrix: covariance is indefinite"));
      }

      if constexpr(triangle_type == TriangleType::diagonal)
      {
        static_assert(diagonal_matrix<A>);
        auto vec = Eigen::Matrix<Scalar, dimensions, 1>::Constant(std::sqrt(s));
        return DiagonalMatrix<decltype(vec)> {vec};
      }
      else if constexpr(triangle_type == TriangleType::lower)
      {
        auto col0 = Eigen::Matrix<Scalar, dimensions, 1>::Constant(std::sqrt(s));
        ZeroMatrix<Scalar, dimensions, dimensions - 1> othercols;
        return TriangularMatrix<M, triangle_type> {concatenate_horizontal(col0, othercols)};
      }
      else
      {
        static_assert(triangle_type == TriangleType::upper);
        auto row0 = Eigen::Matrix<Scalar, 1, dimensions>::Constant(std::sqrt(s));
        ZeroMatrix<Scalar, dimensions - 1, dimensions> otherrows;
        return TriangularMatrix<M, triangle_type> {concatenate_vertical(row0, otherrows)};
      }
    }
    else if constexpr (diagonal_matrix<NestedMatrix>)
    {
      return Cholesky_factor<triangle_type>(nested_matrix(std::forward<A>(a)));
    }
    else if constexpr (MatrixTraits<A>::storage_triangle == TriangleType::diagonal)
    {
      auto n = make_self_contained<A>(nested_matrix(std::forward<A>(a)).diagonal().cwiseSqrt());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }
    else
    {
      // For the general case, perform an LLT Cholesky decomposition.
      M b;
      auto LL_x = a.view().llt();
      if (LL_x.info() == Eigen::Success)
      {
        if constexpr(triangle_type == MatrixTraits<A>::storage_triangle)
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
        auto LDL_x = nested_matrix(std::forward<A>(a)).ldlt();
        if ((not LDL_x.isPositive() and not LDL_x.isNegative()) or LDL_x.info() != Eigen::Success) [[unlikely]]
        {
          if (LDL_x.isPositive() and LDL_x.isNegative()) // Covariance is zero, even though decomposition failed.
          {
            using BM = nested_matrix_t<A>;
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


 /**
  * \brief Take the Cholesky factor of a Eigen::SelfAdjointMatrix.
  * \param a An Eigen::SelfAdjointMatrix.
  * \return Triangle t (upper or lower, depending on a's storage type), where tt<sup>T</sup> = a or t<sup>T</sup>t = a.
  */
#ifdef __cpp_concepts
  template<eigen_self_adjoint_expr A>
#else
  template<typename A, std::enable_if_t<eigen_self_adjoint_expr<A>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(A&& a) noexcept
  {
    return Cholesky_factor<MatrixTraits<A>::storage_triangle>(std::forward<A>(a));
  }


  /**
   * \brief Take the Cholesky square of an Eigen::TriangularMatrix.
   * \param t An Eigen::TriangularMatrix.
   * \return tt<sup>T</sup>
   */
#ifdef __cpp_concepts
  template<eigen_triangular_expr T>
#else
  template<typename T, std::enable_if_t<eigen_triangular_expr<T>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_square(T&& t) noexcept
  {
    constexpr auto triangle_type = MatrixTraits<T>::triangle_type;

    if constexpr(identity_matrix<T> or zero_matrix<T>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<T>>)
    {
      return Cholesky_square(nested_matrix(std::forward<T>(t)));
    }
    else if constexpr (triangle_type == TriangleType::diagonal)
    {
      return DiagonalMatrix {
        make_self_contained<T>(nested_matrix(std::forward<T>(t)).diagonal().array().square().matrix())};
    }
    else
    {
      auto prod {make_native_matrix(adjoint(t))};
      if constexpr (triangle_type == TriangleType::upper)
      {
        prod.applyOnTheRight(std::forward<T>(t).view());
      }
      else
      {
        static_assert(triangle_type == TriangleType::lower);
        prod.applyOnTheLeft(std::forward<T>(t).view());
      }
      return SelfAdjointMatrix<decltype(prod), triangle_type> {std::move(prod)};
    }

  }


  /**
   * \brief Take the Cholesky factor of a diagonal Eigen::TriangularMatrix.
   * \param a A diagonal Eigen::TriangularMatrix.
   * \return A diagonal matrix d, where dd<sup>T</sup> = a.
   */
#ifdef __cpp_concepts
  template<TriangleType = TriangleType::diagonal, eigen_triangular_expr T> requires diagonal_matrix<T>
#else
  template<TriangleType = TriangleType::diagonal, typename T, std::enable_if_t<
    eigen_triangular_expr<T> and diagonal_matrix<T>, int> = 0>
#endif
  constexpr decltype(auto)
  Cholesky_factor(T&& t) noexcept
  {
    if constexpr(identity_matrix<T> or zero_matrix<T>)
    {
      return std::forward<T>(t);
    }
    else if constexpr (diagonal_matrix<nested_matrix_t<T>>)
    {
      return Cholesky_factor(nested_matrix(std::forward<T>(t)));
    }
    else
    {
      static_assert(MatrixTraits<T>::triangle_type == TriangleType::diagonal);
      auto n = make_self_contained<T>(nested_matrix(std::forward<T>(t)).diagonal().cwiseSqrt());
      return DiagonalMatrix<decltype(n)> {std::move(n)};
    }

  }

}


#endif //OPENKALMAN_EIGEN3_CHOLESKY_HPP
