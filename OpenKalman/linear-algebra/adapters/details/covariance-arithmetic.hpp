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

#include "interfaces/eigen/details/eigen-forward-declarations.hpp"
#include "values/math/sqrt.hpp"

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


  namespace detail
  {
    template<typename Arg>
    constexpr decltype(auto) to_nestable(Arg&& arg)
    {
      if constexpr (covariance<Arg>)
      {
        return oin::to_covariance_nestable(std::forward<Arg>(arg));
      }
      else
      {
        static_assert(typed_matrix<Arg>);
        return nested_object(std::forward<Arg>(arg));
      }
    }
  } // namespace detail
  namespace okd = OpenKalman::detail;


  /*
   * (covariance + covariance) or (covariance + typed matrix) or (typed matrix + covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_shaped<Arg2>))) or
      ((typed_matrix<Arg1> and square_shaped<Arg1>) and covariance<Arg2>)) and
    compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_shaped<Arg2>))) or
      ((typed_matrix<Arg1> and square_shaped<Arg1>) and covariance<Arg2>)) and
    compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>, int> = 0>
#endif
  constexpr decltype(auto) operator+(Arg1&& arg1, Arg2&& arg2)
  {
    using C = vector_space_descriptor_of_t<Arg1, 0>;

    if constexpr (zero<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else if constexpr (zero<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (cholesky_form<Arg1> and cholesky_form<Arg2> and
      self_adjoint_covariance<Arg1> and self_adjoint_covariance<Arg2>)
    {
      auto&& e1 = nested_object(std::forward<Arg1>(arg1)); using E1 = decltype(e1);
      auto&& e2 = nested_object(std::forward<Arg2>(arg2)); using E2 = decltype(e2);
      if constexpr (triangular_matrix<E1, TriangleType::upper> and triangular_matrix<E2, TriangleType::upper>)
      {
        return make_covariance<C>(QR_decomposition(concatenate_vertical(std::forward<E1>(e1), std::forward<E2>(e2))));
      }
      else if constexpr (triangular_matrix<E1, TriangleType::upper> and triangular_matrix<E2, TriangleType::lower>)
      {
        return make_covariance<C>(QR_decomposition(concatenate_vertical(std::forward<E1>(e1), adjoint(std::forward<E2>(e2)))));
      }
      else if constexpr (triangular_matrix<E1, TriangleType::lower> and triangular_matrix<E2, TriangleType::upper>)
      {
        return make_covariance<C>(LQ_decomposition(concatenate_horizontal(std::forward<E1>(e1), adjoint(std::forward<E2>(e2)))));
      }
      else
      {
        return make_covariance<C>(LQ_decomposition(concatenate_horizontal(std::forward<E1>(e1), std::forward<E2>(e2))));
      }
    }
    else
    {
      auto&& b1 = okd::to_nestable(std::forward<Arg1>(arg1)); using B1 = decltype(b1);
      auto&& b2 = okd::to_nestable(std::forward<Arg2>(arg2)); using B2 = decltype(b2);

      auto sum = make_self_contained<decltype(b1), decltype(b2)>(std::forward<B1>(b1) + std::forward<B2>(b2));

      if constexpr (hermitian_matrix<decltype(sum)>)
      {
        return make_covariance<C>(std::move(sum));
      }
      else if constexpr (triangular_matrix<decltype(sum)>)
      {
        return make_square_root_covariance<C>(std::move(sum));
      }
      else
      {
        return make_vector_space_adapter(std::move(sum), C{}, C{});
      }
    }
  }


  /*
   * (covariance - covariance) or (covariance - typed matrix) or (typed matrix - covariance)
   */
#ifdef __cpp_concepts
  template<typename Arg1, typename Arg2> requires
  ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_shaped<Arg2>))) or
    ((typed_matrix<Arg1> and square_shaped<Arg1>) and covariance<Arg2>)) and
    coordinate::compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<
    ((covariance<Arg1> and (covariance<Arg2> or (typed_matrix<Arg2> and square_shaped<Arg2>))) or
      ((typed_matrix<Arg1> and square_shaped<Arg1>) and covariance<Arg2>)) and
    compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>, int> = 0>
#endif
  constexpr decltype(auto) operator-(Arg1&& arg1, Arg2&& arg2)
  {
    using C = vector_space_descriptor_of_t<Arg1, 0>;

    if constexpr (zero<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (cholesky_form<Arg1> and cholesky_form<Arg2> and
      self_adjoint_covariance<Arg1> and self_adjoint_covariance<Arg2>)
    {
      using Scalar = scalar_type_of_t<Arg1>;
      using B = nested_object_of_t<Arg2>;

      auto a = nested_object(std::forward<Arg1>(arg1));

      if constexpr (triangular_matrix<B, TriangleType::upper>)
      {
        auto b = to_dense_object(adjoint(nested_object(std::forward<Arg2>(arg2))));
        return make_covariance<C>(make_self_contained(rank_update(std::move(a), std::move(b), Scalar(-1))));
      }
      else
      {
        auto b = to_dense_object(nested_object(std::forward<Arg2>(arg2)));
        return make_covariance<C>(make_self_contained(rank_update(std::move(a), std::move(b), Scalar(-1))));
      }
    }
    else
    {
      auto&& b1 = okd::to_nestable(std::forward<Arg1>(arg1)); using B1 = decltype(b1);
      auto&& b2 = okd::to_nestable(std::forward<Arg2>(arg2)); using B2 = decltype(b2);

      auto diff = make_self_contained<B1, B2>(std::forward<B1>(b1) - std::forward<B2>(b2));

      if constexpr (hermitian_matrix<decltype(diff)>)
      {
        return make_covariance<C>(std::move(diff));
      }
      else if constexpr (triangular_matrix<decltype(diff)>)
      {
        return make_square_root_covariance<C>(std::move(diff));
      }
      else
      {
        return make_vector_space_adapter(std::move(diff), C{}, C{});
      }
    }
  }


  /*
   * covariance * covariance
   */
#ifdef __cpp_concepts
  template<covariance Arg1, covariance Arg2> requires
    coordinate::compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>
#else
  template<typename Arg1, typename Arg2, std::enable_if_t<covariance<Arg1> and covariance<Arg2> and
    compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Arg1&& arg1, Arg2&& arg2)
  {

    if constexpr (zero<Arg1> or identity_matrix<Arg2>)
    {
      return std::forward<Arg1>(arg1);
    }
    else if constexpr (zero<Arg2> or identity_matrix<Arg1>)
    {
      return std::forward<Arg2>(arg2);
    }
    else
    {
      using C = vector_space_descriptor_of_t<Arg1, 0>;

      auto&& b1 = oin::to_covariance_nestable(std::forward<Arg1>(arg1));
      auto&& b2 = oin::to_covariance_nestable(std::forward<Arg2>(arg2));
      using B1 = decltype(b1); using B2 = decltype(b2);

      auto prod = make_self_contained<B1, B2>(std::forward<B1>(b1) * std::forward<B2>(b2));

      if constexpr (hermitian_matrix<decltype(prod)>)
      {
        return make_covariance<C>(std::move(prod));
      }
      else if constexpr (triangular_matrix<decltype(prod)>)
      {
        return make_square_root_covariance<C>(std::move(prod));
      }
      else
      {
        return make_vector_space_adapter(std::move(prod), C{}, C{});
      }
    }
  }


  /*
   * typed matrix * covariance
   */
#ifdef __cpp_concepts
  template<typed_matrix M, covariance Cov> requires
    coordinate::compares_with<vector_space_descriptor_of_t<M, 0>::ColumnCoefficients, typename MatrixTraits<std::decay_t<Cov>>>
#else
  template<typename M, typename Cov, std::enable_if_t<typed_matrix<M> and covariance<Cov> and
    compares_with<vector_space_descriptor_of_t<M, 0>::ColumnCoefficients, typename MatrixTraits<std::decay_t<Cov>>>, int> = 0>
#endif
  constexpr decltype(auto) operator*(M&& m, Cov&& cov)
  {
    using CC = vector_space_descriptor_of_t<Cov, 0>;
    using RC = vector_space_descriptor_of_t<M, 0>;

    if constexpr (zero<M> or zero<Cov>)
    {
      return make_zero(m);
    }
    else if constexpr (identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr (identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr (identity_matrix<nested_object_of_t<M>>)
    {
      return make_vector_space_adapter(oin::to_covariance_nestable(std::forward<Cov>(cov)), RC{}, CC{});
    }
    else
    {
      auto&& mb = nested_object(std::forward<M>(m));
      auto&& cb = oin::to_covariance_nestable(std::forward<Cov>(cov));
      using Mb = decltype(mb); using Cb = decltype(cb);
      auto prod = make_self_contained<Mb, Cb>(std::forward<Mb>(mb) * std::forward<Cb>(cb));
      return Matrix<RC, CC, decltype(prod)> {std::move(prod)};
    }
  }


  /*
   * covariance * typed matrix
   * If the typed matrix is a mean, the result is wrapped.
   */
#ifdef __cpp_concepts
  template<covariance Cov, typed_matrix M> requires
    coordinate::compares_with<vector_space_descriptor_of_t<Cov, 0>, vector_space_descriptor_of_t<M, 0>>
#else
  template<typename Cov, typename M, std::enable_if_t<covariance<Cov> and typed_matrix<M> and
    compares_with<vector_space_descriptor_of_t<Cov, 0>, vector_space_descriptor_of_t<M, 0>>, int> = 0>
#endif
  constexpr decltype(auto) operator*(Cov&& cov, M&& m)
  {
    using RC = vector_space_descriptor_of_t<Cov, 0>;
    using CC = vector_space_descriptor_of_t<M, 1>;

    if constexpr (zero<Cov> or zero<M>)
    {
      return make_zero(m);
    }
    else if constexpr (identity_matrix<M>)
    {
      return std::forward<Cov>(cov);
    }
    else if constexpr (identity_matrix<Cov>)
    {
      return std::forward<M>(m);
    }
    else if constexpr (identity_matrix<nested_object_of_t<M>>)
    {
      return make_vector_space_adapter(oin::to_covariance_nestable(std::forward<Cov>(cov)), RC{}, CC{});
    }
    else
    {
      auto&& cb = oin::to_covariance_nestable(std::forward<Cov>(cov));
      auto&& mb = nested_object(std::forward<M>(m));
      using Cb = decltype(cb); using Mb = decltype(mb);
      auto prod = make_self_contained<Cb, Mb>(std::forward<Cb>(cb) * std::forward<Mb>(mb));
      return Matrix<RC, CC, decltype(prod)> {std::move(prod)};
    }
  }


  /*
   * covariance * scalar
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<scalar_type_of_t<M>> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename scalar_type_of<M>::type>, int> = 0>
#endif
  inline auto operator*(M&& m, const S s)
  {
    using Scalar = const scalar_type_of_t<M>;
    if constexpr (zero<M>)
    {
      return std::forward<M>(m);
    }
    else if constexpr (cholesky_form<M>)
    {
      if constexpr (triangular_covariance<M>)
      {
        auto prod = nested_object(std::forward<M>(m)) * static_cast<Scalar>(s);
        TriangleType t = triangle_type_of_v<nested_object_of_t<M>>;
        return make_self_contained<M>(make_covariance(make_triangular_matrix<t>(std::move(prod))));
      }
      else
      {
        using B = typename MatrixTraits<std::decay_t<nested_object_of_t<M>>>::template TriangularAdapterFrom<>;

        if (s > Scalar(0))
        {
          return MatrixTraits<std::decay_t<M>>::make(B {nested_object(std::forward<M>(m)) * value::sqrt(static_cast<Scalar>(s))});
        }
        else if (s < Scalar(0))
        {
          return MatrixTraits<std::decay_t<M>>::make(B {rank_update(
            make_zero(nested_object(m)),
            to_dense_object(nested_object(std::forward<M>(m))),
            static_cast<Scalar>(s))});
        }
        else
        {
          return MatrixTraits<std::decay_t<M>>::make(B {make_zero(nested_object(m))});
        }
      }
    }
    else
    {
      if constexpr (triangular_covariance<M> and not diagonal_matrix<M>)
      {
        auto prod = nested_object(std::forward<M>(m)) * (static_cast<Scalar>(s) * static_cast<Scalar>(s));
        return make_self_contained<M>(make_covariance(make_hermitian_matrix(std::move(prod))));
      }
      else
      {
        auto prod = nested_object(std::forward<M>(m)) * static_cast<Scalar>(s);
        return make_self_contained<M>(make_covariance(make_hermitian_matrix(std::move(prod))));
      }
    }
  }


  /*
   * scalar * covariance
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<scalar_type_of_t<M>> S>
#else
  template<typename S, typename M, std::enable_if_t<
    std::is_convertible_v<S, typename scalar_type_of<M>::type> and covariance<M>, int> = 0>
#endif
  inline auto operator*(const S s, M&& m)
  {
    return operator*(std::forward<M>(m), s);
  }


  /*
   * covariance / scalar
   */
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<scalar_type_of_t<M>> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename scalar_type_of<M>::type>, int> = 0>
#endif
  constexpr auto operator/(M&& m, const S s)
  {
    using Scalar = scalar_type_of_t<M>;
    if constexpr (cholesky_form<M>)
    {
      if constexpr (triangular_covariance<M>)
      {
        auto ret {nested_object(std::forward<M>(m)) / static_cast<Scalar>(s)};
        TriangleType t = triangle_type_of_v<nested_object_of_t<M>>;
        return make_self_contained<M>(make_covariance(make_triangular_matrix<t>(std::move(ret))));
      }
      else
      {
        using B = typename MatrixTraits<std::decay_t<nested_object_of_t<M>>>::template TriangularAdapterFrom<>;

        if (s > Scalar(0))
        {
          return MatrixTraits<std::decay_t<M>>::make(B {nested_object(std::forward<M>(m)) / value::sqrt(static_cast<Scalar>(s))});
        }
        else if (s < Scalar(0))
        {
          return MatrixTraits<std::decay_t<M>>::make(B {rank_update(
            make_zero(nested_object(m)),
            to_dense_object(nested_object(std::forward<M>(m))),
            1 / static_cast<Scalar>(s))});
        }
        else
        {
          throw (std::runtime_error("operator/(Covariance, Scalar): divide by zero"));
        }
      }
    }
    else
    {
      if constexpr(zero<M>)
      {
        return std::forward<M>(m);
      }
      else if constexpr (triangular_covariance<M> and not diagonal_matrix<M>)
      {
        auto ret {nested_object(std::forward<M>(m)) / (static_cast<Scalar>(s) * static_cast<Scalar>(s))};
        return make_self_contained<M>(make_covariance(make_hermitian_matrix(std::move(ret))));
      }
      else
      {
        auto ret {nested_object(std::forward<M>(m)) / static_cast<Scalar>(s)};
        return make_self_contained<M>(make_covariance(make_hermitian_matrix(std::move(ret))));
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
  constexpr auto operator-(M&& m)
  {
    if constexpr (zero<M>)
    {
      return std::forward<M>(m);
    }
    else
    {
      static_assert(not cholesky_form<M> or triangular_covariance<M>,
        "Cannot negate a Cholesky-form Covariance because the square root would be complex.");
      static_assert(cholesky_form<M> or self_adjoint_covariance<M> or diagonal_matrix<M>,
        "With real numbers, it is impossible to represent the negation of a non-diagonal, non-Cholesky-form "
        "square-root covariance.");
      auto ret {-nested_object(std::forward<M>(m))};
      return MatrixTraits<std::decay_t<M>>::make(make_self_contained<M>(std::move(ret)));
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
  constexpr bool operator==(Arg1&& arg1, Arg2&& arg2)
  {
    if constexpr (compares_with<vector_space_descriptor_of_t<Arg1, 0>, vector_space_descriptor_of_t<Arg2, 0>>)
    {
      return to_dense_object(std::forward<Arg1>(arg1)) == to_dense_object(std::forward<Arg2>(arg2));
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
  constexpr bool operator!=(Arg1&& arg1, Arg2&& arg2)
  {
    return not (std::forward<Arg1>(arg1) == std::forward<Arg2>(arg2));
  }
#endif


  /// Scale a covariance by a factor. Equivalent to multiplication by the square of a scalar.
  /// For a square root covariance, this is equivalent to multiplication by the scalar.
#ifdef __cpp_concepts
  template<covariance M, std::convertible_to<scalar_type_of_t<M>> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename scalar_type_of<M>::type>, int> = 0>
#endif
  inline auto
  scale(M&& m, const S s)
  {
    using Scalar = scalar_type_of_t<M>;
    if constexpr (cholesky_form<M> or (diagonal_matrix<M> and triangular_covariance<M>))
    {
      auto ret {nested_object(std::forward<M>(m)) * s};
      return MatrixTraits<std::decay_t<M>>::make(make_self_contained<M>(std::move(ret)));
    }
    else
    {
      auto ret {nested_object(std::forward<M>(m)) * (static_cast<Scalar>(s) * s)};
      return MatrixTraits<std::decay_t<M>>::make(make_self_contained<M>(std::move(ret)));
    }
  }


  /// Scale a covariance by the inverse of a scalar factor. Equivalent by division by the square of a scalar.
  /// For a square root covariance, this is equivalent to division by the scalar.
#ifdef __cpp_concepts
    template<covariance M, std::convertible_to<scalar_type_of_t<M>> S>
#else
  template<typename M, typename S, std::enable_if_t<
    covariance<M> and std::is_convertible_v<S, typename scalar_type_of<M>::type>, int> = 0>
#endif
  inline auto
  inverse_scale(M&& m, const S s)
  {
    using Scalar = scalar_type_of_t<M>;
    if constexpr (cholesky_form<M> or (diagonal_matrix<M> and triangular_covariance<M>))
    {
      auto ret {nested_object(std::forward<M>(m)) / s};
      return MatrixTraits<std::decay_t<M>>::make(make_self_contained<M>(std::move(ret)));
    }
    else
    {
      auto ret {nested_object(std::forward<M>(m)) / (static_cast<Scalar>(s) * s)};
      return MatrixTraits<std::decay_t<M>>::make(make_self_contained<M>(std::move(ret)));
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
    compares_with<vector_space_descriptor_of_t<A, 0>::ColumnCoefficients, typename MatrixTraits<std::decay_t<M>>>and
    (not euclidean_transformed<A>)
#else
  template<typename M, typename A, std::enable_if_t<covariance<M> and typed_matrix<A> and
    compares_with<vector_space_descriptor_of_t<A, 0>::ColumnCoefficients, typename MatrixTraits<std::decay_t<M>>>and
    (not euclidean_transformed<A>), int> = 0>
#endif
  inline auto
  scale(M&& m, A&& a)
  {
    using AC = vector_space_descriptor_of_t<A, 0>;
    using NestedMatrix = nested_object_of_t<M>;

    if constexpr (diagonal_matrix<NestedMatrix> or hermitian_matrix<NestedMatrix>)
    {
      using SABaseType = std::conditional_t<
        diagonal_matrix<NestedMatrix>,
        typename MatrixTraits<std::decay_t<NestedMatrix>>::template SelfAdjointMatrixFrom<TriangleType::lower>,
        typename MatrixTraits<std::decay_t<NestedMatrix>>::template SelfAdjointMatrixFrom<>>;

      if constexpr(triangular_covariance<M>)
      {
        auto b = make_self_contained<M, A>(nested_object(a * (square(std::forward<M>(m)) * adjoint(a))));
        return make_square_root_covariance<AC>(MatrixTraits<std::decay_t<SABaseType>>::make(std::move(b)));
      }
      else
      {
        auto b = make_self_contained<M, A>(nested_object(a * (std::forward<M>(m) * adjoint(a))));
        return make_covariance<AC>(MatrixTraits<std::decay_t<SABaseType>>::make(std::move(b)));
      }
    }
    else if constexpr (triangular_matrix<NestedMatrix, TriangleType::upper>)
    {
      auto b = QR_decomposition(nested_object(std::forward<M>(m)) * adjoint(nested_object(std::forward<A>(a))));
      return MatrixTraits<std::decay_t<M>>::template make<AC>(make_self_contained<M, A>(std::move(b)));
    }
    else
    {
      static_assert(triangular_matrix<NestedMatrix, TriangleType::lower>);
      auto b = LQ_decomposition(nested_object(std::forward<A>(a)) * nested_object(std::forward<M>(m)));
      return MatrixTraits<std::decay_t<M>>::template make<AC>(make_self_contained<M, A>(std::move(b)));
    }
  }


}

#endif //OPENKALMAN_COVARIANCEARITHMETIC_HPP
