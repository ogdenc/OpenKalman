/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEARITHMETIC_HPP
#define OPENKALMAN_COVARIANCEARITHMETIC_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename Arg>
    constexpr decltype(auto) to_nestable(Arg&& arg)
    {
      if constexpr (covariance<Arg>)
      {
        return internal::to_covariance_nestable(std::forward<Arg>(arg));
      }
      else
      {
        static_assert(typed_matrix<Arg>);
        return nested_matrix(std::forward<Arg>(arg));
      }
    }
  } // namespace detail


  /*
   * (covariance + covariance) or (covariance + typed matrix) or (typed matrix + covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_matrix<Arg2>))) or
      ((typed_matrix<Arg1> and square_matrix<Arg1>) and covariance<Arg2>)) and
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_matrix<Arg2>))) or
      ((typed_matrix<Arg1> and square_matrix<Arg1>) and covariance<Arg2>)) and
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    using C = typename MatrixTraits<Arg1>::RowCoefficients;

    if constexpr (zero_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (cholesky_form<Arg1> and cholesky_form<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      decltype(auto) e1 = nested_matrix(std::forward<Arg1>(arg1)); using E1 = decltype(e1);
      decltype(auto) e2 = nested_matrix(std::forward<Arg2>(arg2)); using E2 = decltype(e2);
      if constexpr (upper_triangular_matrix<E1> and upper_triangular_matrix<E2>)
      {
        return make_covariance<C>(make_self_contained<E1,E2>(QR_decomposition(concatenate_vertical(e1, e2))));
      }
      else if constexpr (upper_triangular_matrix<E1> and lower_triangular_matrix<E2>)
      {
        return make_covariance<C>(make_self_contained<E1,E2>(QR_decomposition(concatenate_vertical(e1, adjoint(e2)))));
      }
      else if constexpr (lower_triangular_matrix<E1> and upper_triangular_matrix<E2>)
      {
        return make_covariance<C>(make_self_contained<E1,E2>(LQ_decomposition(concatenate_horizontal(e1, adjoint(e2)))));
      }
      else
      {
        return make_covariance<C>(make_self_contained<E1,E2>(LQ_decomposition(concatenate_horizontal(e1, e2))));
      }
    }
    else
    {
      decltype(auto) b1 = detail::to_nestable(std::forward<Arg1>(arg1));
      decltype(auto) b2 = detail::to_nestable(std::forward<Arg2>(arg2));
      auto sum = make_self_contained<decltype(b1), decltype(b2)>(b1 + b2);
      if constexpr (self_adjoint_matrix<decltype(sum)>)
      {
        return make_covariance<C>(std::move(sum));
      }
      else if constexpr (triangular_matrix<decltype(sum)>)
      {
        return make_square_root_covariance<C>(std::move(sum));
      }
      else
      {
        return make_matrix<C, C>(std::move(sum));
      }
    }
  }


  /*
   * (covariance - covariance) or (covariance - typed matrix) or (typed matrix - covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
  ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_matrix<Arg2>))) or
    ((typed_matrix<Arg1> and square_matrix<Arg1>) and covariance<Arg2>)) and
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_matrix<Arg2>))) or
      ((typed_matrix<Arg1> and square_matrix<Arg1>) and covariance<Arg2>)) and
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    using C = typename MatrixTraits<Arg1>::RowCoefficients;

    if constexpr (zero_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (cholesky_form<Arg1> and cholesky_form<Arg2> and
      not square_root_covariance<Arg1> and not square_root_covariance<Arg2>)
    {
      using Scalar = typename MatrixTraits<Arg1>::Scalar;
      using B = nested_matrix_t<Arg2>;

      const auto a = nested_matrix(std::forward<Arg1>(arg1));
      const auto b = upper_triangular_matrix<B> ?
        make_native_matrix(adjoint(nested_matrix(std::forward<Arg2>(arg2)))) :
        make_native_matrix(nested_matrix(std::forward<Arg2>(arg2)));

      return make_covariance<C>(make_self_contained(rank_update(std::move(a), std::move(b), Scalar(-1))));
    }
    else
    {
      decltype(auto) b1 = detail::to_nestable(std::forward<Arg1>(arg1));
      decltype(auto) b2 = detail::to_nestable(std::forward<Arg2>(arg2));
      auto diff = make_self_contained<decltype(b1), decltype(b2)>(b1 - b2);
      if constexpr (self_adjoint_matrix<decltype(diff)>)
      {
        return make_covariance<C>(std::move(diff));
      }
      else if constexpr (triangular_matrix<decltype(diff)>)
      {
        return make_square_root_covariance<C>(std::move(diff));
      }
      else
      {
        return make_matrix<C, C>(std::move(diff));
      }
    }
  }


  /*
   * covariance * covariance
   */
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2> requires
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2> and
    equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2) noexcept
  {

    if constexpr (zero_matrix<Arg1> or zero_matrix<Arg2>)
    {
      return MatrixTraits<Arg1>::zero();
    }
    else if constexpr (identity_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (identity_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else
    {
      using C = typename MatrixTraits<Arg1>::RowCoefficients;
      decltype(auto) b1 = internal::to_covariance_nestable(std::forward<Arg1>(arg1));
      decltype(auto) b2 = internal::to_covariance_nestable(std::forward<Arg2>(arg2));
      auto prod = make_self_contained<decltype(b1), decltype(b2)>(b1 * b2);
      if constexpr (self_adjoint_matrix<decltype(prod)>)
      {
        return make_covariance<C>(std::move(prod));
      }
      else if constexpr (triangular_matrix<decltype(prod)>)
      {
        return make_square_root_covariance<C>(std::move(prod));
      }
      else
      {
        return make_matrix<C, C>(std::move(prod));
      }
    }
  }


  /*
   * typed matrix * covariance
   */
#ifdef __cpp_concepts
  template<typed_matrix M, covariance Cov> requires
    equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, typename MatrixTraits<Cov>::RowCoefficients>
#else
  template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and covariance<Cov> and
    equivalent_to<typename MatrixTraits<M>::ColumnCoefficients, typename MatrixTraits<Cov>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator*(M&& m, Cov&& cov) noexcept
  {
    using CC = typename MatrixTraits<Cov>::RowCoefficients;
    using RC = typename MatrixTraits<M>::RowCoefficients;
    using Mat = native_matrix_t<M, RC::dimensions, CC::dimensions>;

    if constexpr (zero_matrix<M> or zero_matrix<Cov>)
    {
      return make_matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr (identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr (identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr (identity_matrix<nested_matrix_t<M>>)
    {
      return make_matrix<RC, CC>(internal::to_covariance_nestable(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) mb = nested_matrix(std::forward<M>(m));
      decltype(auto) cb = internal::to_covariance_nestable(std::forward<Cov>(cov));
      auto prod = make_self_contained<decltype(mb), decltype(cb)>(mb * cb);
      return make_matrix<RC, CC>(std::move(prod));
    }
  }


  /*
   * covariance * typed matrix
   * If the typed matrix is a mean, the result is wrapped.
   */
#ifdef __cpp_concepts
  template<covariance Cov, typed_matrix M> requires
    equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, typename MatrixTraits<M>::RowCoefficients>
#else
  template<typename Cov, typename M, std::enable_if_t<covariance<Cov> and typed_matrix<M> and
    equivalent_to<typename MatrixTraits<Cov>::RowCoefficients, typename MatrixTraits<M>::RowCoefficients>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Cov&& cov, M&& m) noexcept
  {
    using RC = typename MatrixTraits<Cov>::RowCoefficients;
    using CC = typename MatrixTraits<M>::ColumnCoefficients;
    using Mat = native_matrix_t<M, RC::dimensions, CC::dimensions>;

    if constexpr (zero_matrix<Cov> or zero_matrix<M>)
    {
      return make_matrix<RC, CC>(MatrixTraits<Mat>::zero());
    }
    else if constexpr (identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr (identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr (identity_matrix<nested_matrix_t<M>>)
    {
      return make_matrix<RC, CC>(internal::to_covariance_nestable(std::forward<Cov>(cov)));
    }
    else
    {
      decltype(auto) cb = internal::to_covariance_nestable(std::forward<Cov>(cov));
      decltype(auto) mb = nested_matrix(std::forward<M>(m));
      auto prod = make_self_contained<decltype(cb), decltype(mb)>(cb * mb);
      return make_matrix<RC, CC>(std::move(prod));
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
  inline auto operator*(M&& m, const S s) noexcept
  {
    using Scalar = const typename MatrixTraits<M>::Scalar;
    if constexpr (cholesky_form<M>)
    {
      if constexpr (square_root_covariance<M>)
      {
        auto prod = nested_matrix(std::forward<M>(m)) * static_cast<Scalar>(s);
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(prod)));
      }
      else
      {
        auto b = make_self_contained(nested_matrix(std::forward<M>(m))); using B = decltype(b);
        if (s > Scalar(0))
        {
          b *= std::sqrt(static_cast<Scalar>(s));
        }
        else if (s < Scalar(0))
        {
          const auto u = make_native_matrix(b);
          b = MatrixTraits<B>::zero();
          rank_update(b, u, static_cast<Scalar>(s));
        }
        else
        {
          b = MatrixTraits<B>::zero();
        }
        return MatrixTraits<M>::make(std::move(b));
      }
    }
    else
    {
      if constexpr (zero_matrix<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else if constexpr (square_root_covariance<M> and not diagonal_matrix<M>)
      {
        auto prod = nested_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * static_cast<Scalar>(s));
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(prod)));
      }
      else
      {
        auto prod = nested_matrix(std::forward<M>(m)) * static_cast<Scalar>(s);
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(prod)));
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
  inline auto operator*(const S s, M&& m) noexcept
  {
    return operator*(std::forward<M>(m), s);
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
    if constexpr (cholesky_form<M>)
    {
      if constexpr (square_root_covariance<M>)
      {
        auto ret = nested_matrix(std::forward<M>(m)) / static_cast<Scalar>(s);
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
      }
      else
      {
        auto b = make_self_contained(nested_matrix(std::forward<M>(m)));
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
      else if constexpr (square_root_covariance<M> and not diagonal_matrix<M>)
      {
        auto ret = nested_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * static_cast<Scalar>(s));
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
      }
      else
      {
        auto ret = nested_matrix(std::forward<M>(m)) / static_cast<Scalar>(s);
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
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
      if constexpr (square_root_covariance<M>)
      {
        auto ret = -nested_matrix(std::forward<M>(m));
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
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
      if constexpr (zero_matrix<M>)
      {
        return MatrixTraits<M>::zero();
      }
      else
      {
        static_assert(not square_root_covariance<M> or diagonal_matrix<M>,
          "With real numbers, it is impossible to represent the negation of a non-diagonal, non-Cholesky-form "
          "square-root covariance.");
        auto ret = -nested_matrix(std::forward<M>(m));
        return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
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
    if constexpr (std::is_same_v<native_matrix_t<Arg1>, native_matrix_t<Arg2>> and
      equivalent_to<typename MatrixTraits<Arg1>::RowCoefficients, typename MatrixTraits<Arg2>::RowCoefficients>)
    {
      return make_native_matrix(arg1) == make_native_matrix(arg2);
    }
    else
    {
      return false;
    }
  }


#ifndef __cpp_impl_three_way_comparison
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
#endif


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
    if constexpr (cholesky_form<M> or (diagonal_matrix<M> and square_root_covariance<M>))
    {
      auto ret = nested_matrix(std::forward<M>(m)) * s;
      return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
    }
    else
    {
      auto ret = nested_matrix(std::forward<M>(m)) * (static_cast<Scalar>(s) * s);
      return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
    }
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
    if constexpr (cholesky_form<M> or (diagonal_matrix<M> and square_root_covariance<M>))
    {
      auto ret = nested_matrix(std::forward<M>(m)) / s;
      return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
    }
    else
    {
      auto ret = nested_matrix(std::forward<M>(m)) / (static_cast<Scalar>(s) * s);
      return MatrixTraits<M>::make(make_self_contained<M>(std::move(ret)));
    }
  }


  /**
   * \brief Scale a covariance by a matrix.
   * \tparam M A \ref covariance.
   * \tparam A A \ref typed_matrix.
   * \details A scaled covariance Arg is A * Arg * adjoint(A).
   * A scaled square root covariance L or U is also scaled accordingly, so that
   * scale(L * adjoint(L)) = A * L * adjoint(L) * adjoint(A) or
   * scale(adjoint(U) * U) = A * adjoint(U) * U * adjoint(A).
   */
#ifdef __cpp_concepts
  template<covariance M, typed_matrix A> requires
    equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, typename MatrixTraits<M>::RowCoefficients> and
    (not euclidean_transformed<A>)
#else
  template<typename M, typename A, std::enable_if_t<covariance<M> and typed_matrix<A> and
    equivalent_to<typename MatrixTraits<A>::ColumnCoefficients, typename MatrixTraits<M>::RowCoefficients> and
    (not euclidean_transformed<A>), int> = 0>
#endif
  inline auto
  scale(M&& m, A&& a)
  {
    using AC = typename MatrixTraits<A>::RowCoefficients;
    using NestedMatrix = nested_matrix_t<M>;

    if constexpr (diagonal_matrix<NestedMatrix> or self_adjoint_matrix<NestedMatrix>)
    {
      using SABaseType = std::conditional_t<
        diagonal_matrix<NestedMatrix>,
        typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<TriangleType::lower>,
        typename MatrixTraits<NestedMatrix>::template SelfAdjointMatrixFrom<>>;

      if constexpr(square_root_covariance<M>)
      {
        auto b = make_self_contained<M, A>(nested_matrix(a * (square(m) * adjoint(a))));
        return make_square_root_covariance<AC>(MatrixTraits<SABaseType>::make(std::move(b)));
      }
      else
      {
        auto b = make_self_contained<M, A>(nested_matrix(a * (m * adjoint(a))));
        return make_covariance<AC>(MatrixTraits<SABaseType>::make(std::move(b)));
      }
    }
    else if constexpr (upper_triangular_matrix<NestedMatrix>)
    {
      auto b = QR_decomposition(nested_matrix(m) * adjoint(nested_matrix(a)));
      return MatrixTraits<M>::template make<AC>(make_self_contained<M, A>(std::move(b)));
    }
    else
    {
      static_assert(lower_triangular_matrix<NestedMatrix>);
      auto b = LQ_decomposition(nested_matrix(a) * nested_matrix(m));
      return MatrixTraits<M>::template make<AC>(make_self_contained<M, A>(std::move(b)));
    }
  }


}

#endif //OPENKALMAN_COVARIANCEARITHMETIC_HPP
