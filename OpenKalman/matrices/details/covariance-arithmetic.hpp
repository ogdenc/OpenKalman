/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2020 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEARITHMETIC_H
#define OPENKALMAN_COVARIANCEARITHMETIC_H

namespace OpenKalman
{
  /*
   * (covariance + covariance) or (covariance + typed matrix) or (typed matrix + covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (covariance<Arg1> and (covariance<Arg2> or typed_matrix<Arg2>)) or
    (typed_matrix<Arg1> and covariance<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (covariance<Arg1> and (covariance<Arg2> or typed_matrix<Arg2>)) or
    (typed_matrix<Arg1> and covariance<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    using Cov = std::conditional_t<covariance<Arg1>, Arg1, Arg2>;
    using Other = std::conditional_t<covariance<Arg1>, Arg2, Arg1>;
    using C = typename MatrixTraits<Cov>::Coefficients;

    if constexpr(typed_matrix<Other>)
    {
      static_assert(
        equivalent_to<C, typename MatrixTraits<Other>::RowCoefficients> and
        equivalent_to<C, typename MatrixTraits<Other>::ColumnCoefficients>);
    }
    else
    {
      static_assert(equivalent_to<C, typename MatrixTraits<Arg2>::Coefficients>);
    }

    if constexpr(zero_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(cholesky_form<Arg1> and cholesky_form<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      decltype(auto) E1 = base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) E2 = base_matrix(std::forward<Arg2>(arg2));
      if constexpr(upper_triangular_matrix<decltype(E1)> and upper_triangular_matrix<decltype(E2)>)
      {
        return make_Covariance<C>(make_self_contained(QR_decomposition(concatenate_vertical(E1, E2))));
      }
      else if constexpr(upper_triangular_matrix<decltype(E1)> and lower_triangular_matrix<decltype(E2)>)
      {
        return make_Covariance<C>(make_self_contained(QR_decomposition(concatenate_vertical(E1, adjoint(E2)))));
      }
      else if constexpr(lower_triangular_matrix<decltype(E1)> and upper_triangular_matrix<decltype(E2)>)
      {
        return make_Covariance<C>(make_self_contained(LQ_decomposition(concatenate_horizontal(E1, adjoint(E2)))));
      }
      else
      {
        return make_Covariance<C>(make_self_contained(LQ_decomposition(concatenate_horizontal(E1, E2))));
      }
    }
    else
    {
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      auto sum = make_self_contained<decltype(b1), decltype(b2)>(b1 + b2);
      if constexpr(self_adjoint_matrix<decltype(sum)>)
      {
        return make_Covariance<C>(std::move(sum));
      }
      else if constexpr(triangular_matrix<decltype(sum)>)
      {
        return make_SquareRootCovariance<C>(std::move(sum));
      }
      else
      {
        return make_Matrix<C, C>(std::move(sum));
      }
    }
  }


  /*
   * (covariance - covariance) or (covariance - typed matrix) or (typed matrix - covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    (covariance<Arg1> and (covariance<Arg2> or typed_matrix<Arg2>)) or
    (typed_matrix<Arg1> and covariance<Arg2>)
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    (covariance<Arg1> and (covariance<Arg2> or typed_matrix<Arg2>)) or
    (typed_matrix<Arg1> and covariance<Arg2>), int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    using Cov = std::conditional_t<covariance<Arg1>, Arg1, Arg2>;
    using Other = std::conditional_t<covariance<Arg1>, Arg2, Arg1>;
    using C = typename MatrixTraits<Cov>::Coefficients;

    if constexpr(typed_matrix<Other>)
    {
      static_assert(equivalent_to<C, typename MatrixTraits<Other>::RowCoefficients> and
        equivalent_to<C, typename MatrixTraits<Other>::ColumnCoefficients>);
    }
    else
    {
      static_assert(equivalent_to<C, typename MatrixTraits<Arg2>::Coefficients>);
    }

    if constexpr(zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr(cholesky_form<Arg1> and cholesky_form<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      using Scalar = typename MatrixTraits<Arg1>::Scalar;
      using B = typename MatrixTraits<Arg2>::BaseMatrix;

      decltype(auto) a = base_matrix(std::forward<Arg1>(arg1));
      const auto b = upper_triangular_matrix<B> ?
        make_native_matrix(adjoint(base_matrix(std::forward<Arg2>(arg2)))) :
        make_native_matrix(base_matrix(std::forward<Arg2>(arg2)));

      return make_Covariance<C>(make_self_contained(rank_update(a, b, Scalar(-1))));
    }
    else
    {
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      auto diff = make_self_contained<decltype(b1), decltype(b2)>(b1 - b2);
      if constexpr(self_adjoint_matrix<decltype(diff)>)
      {
        return make_Covariance<C>(std::move(diff));
      }
      else if constexpr(triangular_matrix<decltype(diff)>)
      {
        return make_SquareRootCovariance<C>(std::move(diff));
      }
      else
      {
        return make_Matrix<C, C>(std::move(diff));
      }
    }
  }


  /*
   * covariance * covariance
   */
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2> requires
    equivalent_to<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2> and
    equivalent_to<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2) noexcept
  {

    if constexpr(zero_matrix<Arg1> or zero_matrix<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr(identity_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr(identity_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      using C = typename MatrixTraits<Arg1>::Coefficients;
      decltype(auto) b1 = internal::convert_base_matrix(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::convert_base_matrix(std::forward<Arg2>(arg2));
      auto prod = make_self_contained<decltype(b1), decltype(b2)>(b1 * b2);
      if constexpr(self_adjoint_matrix<decltype(prod)>)
      {
        return make_Covariance<C>(std::move(prod));
      }
      else if constexpr(triangular_matrix<decltype(prod)>)
      {
        return make_SquareRootCovariance<C>(std::move(prod));
      }
      else
      {
        return make_Matrix<C, C>(std::move(prod));
      }
    }
  }


  /*
   * typed matrix * covariance
   */
#ifdef __cpp_concepts
  template<typed_matrix M, covariance Cov>
#else
  template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and covariance<Cov>, int> = 0>
#endif
  constexpr decltype(auto) operator*(M&& m, Cov&& cov) noexcept
  {
    using CC = typename MatrixTraits<Cov>::Coefficients;
    static_assert(equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, CC>);
    using RC = typename MatrixTraits<M>::RowCoefficients;
    using Mat = native_matrix_t<M, RC::size, CC::size>;

    if constexpr(zero_matrix<M> or zero_matrix<Cov>)
    {
      return make_Matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr(identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr(identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr(identity_matrix<typename MatrixTraits<M>::BaseMatrix>)
    {
      return make_Matrix<RC, CC>(internal::convert_base_matrix(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) mb = base_matrix(std::forward<M>(m));
      decltype(auto) cb = internal::convert_base_matrix(std::forward<Cov>(cov));
      auto prod = make_self_contained<decltype(mb), decltype(cb)>(mb * cb);
      return make_Matrix<RC, CC>(std::move(prod));
    }
  }


  /*
   * covariance * typed matrix
   * If the typed matrix is a mean, the result is wrapped.
   */
#ifdef __cpp_concepts
  template<covariance Cov, typed_matrix M>
#else
  template<typename Cov, typename M, std::enable_if_t<covariance<Cov> and typed_matrix<M>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Cov&& cov, M&& m) noexcept
  {
    using RC = typename MatrixTraits<Cov>::Coefficients;
    static_assert(equivalent_to<RC, typename MatrixTraits<M>::RowCoefficients>);
    using CC = typename MatrixTraits<M>::ColumnCoefficients;
    using Mat = native_matrix_t<M, RC::size, CC::size>;

    if constexpr(zero_matrix<Cov> or zero_matrix<M>)
    {
      return make_Matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr(identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr(identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr(identity_matrix<typename MatrixTraits<M>::BaseMatrix>)
    {
      return make_Matrix<RC, CC>(internal::convert_base_matrix(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) cb = internal::convert_base_matrix(std::forward<Cov>(cov));
      decltype(auto) mb = base_matrix(std::forward<M>(m));
      auto prod = make_self_contained<decltype(cb), decltype(mb)>(cb * mb);
      return make_Matrix<RC, CC>(std::move(prod));
    }
  }


  /*
   * covariance * scalar
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  constexpr auto operator*(M&& m, const S s) noexcept
  {
    using Scalar = const typename MatrixTraits<M>::Scalar;
    if constexpr(cholesky_form<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * static_cast<Scalar>(s));
        return make_self_contained<M>(std::move(ret));
      }
      else
      {
        auto b = make_self_contained(base_matrix(std::forward<M>(m)));
        if (s > Scalar(0))
        {
          b *= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < Scalar(0))
        {
          const auto u = make_native_matrix(b);
          b = MatrixTraits<decltype(b)>::zero();
          rank_update(b, u, static_cast<Scalar>(s));
        }
        else
        {
          b = MatrixTraits<decltype(b)>::zero();
        }
        return MatrixTraits<M>::make(std::move(b));
      }
    }
    else
    {
      if constexpr(zero_matrix<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else if constexpr(square_root_covariance<M> and not diagonal_matrix<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * static_cast<Scalar>(s)));
        return make_self_contained<M>(std::move(ret));
      }
      else
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * static_cast<Scalar>(s));
        return make_self_contained<M>(std::move(ret));
      }
    }
  }


  /*
   * scalar * covariance
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename S, typename M, std::enable_if_t<
    std::is_convertible_v<S, typename MatrixTraits<M>::Scalar> and covariance<M>, int> = 0>
#endif
  constexpr auto operator*(const S s, M&& m) noexcept
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    auto ret = std::forward<M>(m) * static_cast<Scalar>(s);
    return make_self_contained<M>(std::move(ret));
  }


  /*
   * covariance / scalar
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  constexpr auto operator/(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(cholesky_form<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / static_cast<Scalar>(s));
        return make_self_contained<M>(std::move(ret));
      }
      else
      {
        auto b = make_self_contained(base_matrix(std::forward<M>(m)));
        if (s > Scalar(0))
        {
          b /= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < Scalar(0))
        {
          const auto u = make_native_matrix(b);
          b = MatrixTraits<decltype(b)>::zero();
          rank_update(b, u, 1 / static_cast<Scalar>(s));
        }
        else
        {
          throw (std::runtime_error("operator/(Covariance, Scalar): divide by zero"));
        }
        return MatrixTraits<M>::make(std::move(b));
      }
    }
    else
    {
      if constexpr(zero_matrix<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else if constexpr(square_root_covariance<M> and not diagonal_matrix<M>)
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * static_cast<Scalar>(s)));
        return make_self_contained<M>(std::move(ret));
      }
      else
      {
        auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / static_cast<Scalar>(s));
        return make_self_contained<M>(std::move(ret));
      }
    }
  }


  /*
   * negation of a covariance
   */
#ifdef __cpp_concepts
  template<covariance M>
#else
  template<typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  constexpr auto operator-(M&& m) noexcept
  {
    static_assert(not cholesky_form<M> or square_root_covariance<M>,
      "Cannot negate a Cholesky-based Covariance because the square root would be complex.");
    if constexpr(cholesky_form<M>)
    {
      if constexpr(square_root_covariance<M>)
      {
        auto ret = MatrixTraits<M>::make(-base_matrix(std::forward<M>(m)));
        return make_self_contained<M>(std::move(ret));
      }
      else
      {
        auto res = make_self_contained(std::forward<M>(m));
        res *= MatrixTraits<M>::Scalar(-1);
        return res;
      }
    }
    else
    {
      if constexpr(zero_matrix<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else
      {
        static_assert(not square_root_covariance<M> or diagonal_matrix<M>,
          "With real numbers, it is impossible to represent the negation of a non-diagonal, non-Cholesky-form "
          "square-root covariance.");
        auto ret = MatrixTraits<M>::make(-base_matrix(std::forward<M>(m)));
        return make_self_contained<M>(std::move(ret));
      }
    }
  }


  /*
   * covariance == covariance
   */
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2>, int> = 0>
#endif
  constexpr bool operator==(const Arg1& arg1, const Arg2& arg2)
  {
    if constexpr(std::is_same_v<native_matrix_t<Arg1>, native_matrix_t<Arg2>> and
      equivalent_to<typename MatrixTraits<Arg1>::Coefficients, typename MatrixTraits<Arg2>::Coefficients>)
    {
      return make_native_matrix(arg1) == make_native_matrix(arg2);
    }
    else
    {
      return false;
    }
  }


  /*
   * covariance != covariance
   */
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2>, int> = 0>
#endif
  constexpr bool operator!=(const Arg1& arg1, const Arg2& arg2)
  {
    return not (arg1 == arg2);
  }


  /// Scale a covariance by a factor. Equivalent to multiplication by the square of a scalar.
  /// For a square root covariance, this is equivalent to multiplication by the scalar.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  inline auto
  scale(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(cholesky_form<M> or (diagonal_matrix<M> and square_root_covariance<M>))
      return MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * s);
    else
      return MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * s));
  }


  /// Scale a covariance by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
  /// For a square root covariance, this is equivalent to division by the scalar.
#ifdef __cpp_concepts
    template<covariance M, std::convertible_to<typename MatrixTraits<M>::Scalar> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename MatrixTraits<M>::Scalar>, int> = 0>
#endif
  inline auto
  inverse_scale(M&& m, const S s)
  {
    using Scalar = typename MatrixTraits<M>::Scalar;
    if constexpr(cholesky_form<M> or (diagonal_matrix<M> and square_root_covariance<M>))
    {
      auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / s);
      return make_self_contained<M>(std::move(ret));
    }
    else
    {
      auto ret = MatrixTraits<M>::make(base_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * s));
      return make_self_contained<M>(std::move(ret));
    }
  }


  /// Scale a covariance by a matrix.
  /// A scaled covariance Arg is A * Arg * adjoint(A).
  /// A scaled square root covariance L or U is also scaled accordingly, so that
  /// scale(L * adjoint(L)) = A * L * adjoint(L) * adjoint(A) or
  /// scale(adjoint(U) * U) = A * adjoint(U) * U * adjoint(A).
#ifdef __cpp_concepts
  template<covariance M, typed_matrix A>
#else
  template<typename M, typename A, std::enable_if_t<covariance<M> and typed_matrix<A>, int> = 0>
#endif
  inline auto
  scale(M&& m, A&& a)
  {
    using C = typename MatrixTraits<M>::Coefficients;
    using AC = typename MatrixTraits<A>::RowCoefficients;
    static_assert(equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, C>);
    static_assert(not euclidean_transformed<A>);
    using BaseMatrix = typename MatrixTraits<M>::BaseMatrix;

    decltype(auto) mbase = base_matrix(std::forward<M>(m));
    decltype(auto) abase = base_matrix(std::forward<A>(a));

    if constexpr(diagonal_matrix<BaseMatrix>)
    {
      using SABaseType = typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<TriangleType::lower>;
      if constexpr(square_root_covariance<M>)
      {
        auto b = MatrixTraits<SABaseType>::make(make_self_contained(abase * (Cholesky_square(mbase) * adjoint(abase))));
        return make_SquareRootCovariance<AC>(std::move(b));
      }
      else
      {
        auto b = MatrixTraits<SABaseType>::make(make_self_contained(abase * (mbase * adjoint(abase))));
        return make_Covariance<AC>(std::move(b));
      }
    }
    else if constexpr(self_adjoint_matrix<BaseMatrix>)
    {
      using SABaseType = typename MatrixTraits<BaseMatrix>::template SelfAdjointBaseType<>;
      auto b = make_self_contained(MatrixTraits<SABaseType>::make(abase * (mbase * adjoint(abase))));
      return MatrixTraits<M>::template make<AC>(std::move(b));
    }
    else if constexpr(upper_triangular_matrix<BaseMatrix>)
    {
      const auto b = mbase * adjoint(abase);
      return MatrixTraits<M>::template make<AC>(make_self_contained(QR_decomposition(std::move(b))));
    }
    else
    {
      const auto b = abase * base_matrix(std::forward<M>(m));
      return MatrixTraits<M>::template make<AC>(make_self_contained(LQ_decomposition(std::move(b))));
    }
  }


}

#endif //OPENKALMAN_COVARIANCEARITHMETIC_H
