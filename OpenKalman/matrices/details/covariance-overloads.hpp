/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2017-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_COVARIANCEOVERLOADS_HPP
#define OPENKALMAN_COVARIANCEOVERLOADS_HPP

#include <iostream>

namespace OpenKalman
{
  namespace oin = OpenKalman::internal;


#ifdef __cpp_concepts
  template<self_adjoint_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<self_adjoint_covariance<Arg>, int> = 0>
#endif
  inline auto
  square_root(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).square_root();
  }


#ifdef __cpp_concepts
  template<triangular_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<triangular_covariance<Arg>, int> = 0>
#endif
  inline auto
  square(Arg&& arg) noexcept
  {
    return std::forward<Arg>(arg).square();
  }

} // namespace OpenKalman


namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<covariance T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (not self_adjoint_covariance<T> or element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), I...>) and
    (not triangular_covariance<T> or element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), I...>)
  struct GetElement<T, I...>
#else
  template<typename T, typename...I>
  struct GetElement<T, I..., std::enable_if_t<covariance<T> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not self_adjoint_covariance<T> or element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), I...>) and
    (not triangular_covariance<T> or element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), I...>)>>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I...i)
    {
      return std::forward<Arg>(arg)(i...);
    }
  };


#ifdef __cpp_concepts
  template<covariance T, std::convertible_to<const std::size_t&>...I> requires (sizeof...(I) <= 2) and
    (not self_adjoint_covariance<T> or element_settable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), I...>) and
    (not triangular_covariance<T> or element_settable<decltype(std::declval<T>().get_triangular_nested_matrix()), I...>)
  struct SetElement<T, I...>
#else
  template<typename T, typename...I>
  struct SetElement<T, I..., std::enable_if_t<covariance<T> and element_settable<nested_matrix_of_t<Arg>, I...> and
    ((sizeof...(I) <= 2) and ... and std::is_convertible_v<I, const std::size_t&>) and
    (not self_adjoint_covariance<T> or element_gettable<decltype(std::declval<T>().get_self_adjoint_nested_matrix()), I...>) and
    (not triangular_covariance<T> or element_gettable<decltype(std::declval<T>().get_triangular_nested_matrix()), I...>)>>
#endif
  {
    template<typename Arg, typename Scalar>
    static constexpr void set(Arg&& arg, Scalar s, I...i)
    {
      arg.set_element(s, i...);
    }
  };

} // namespace OpenKalman::interface


namespace OpenKalman
{

  namespace interface
  {

#ifdef __cpp_concepts
  template<covariance T>
  struct Subsets<T>
#else
  template<typename T>
  struct Subsets<T, std::enable_if_t<covariance<T>>>
#endif
  {
    // \todo Add come of this logic to global functions.

    template<std::size_t...index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    column(Arg&& arg, runtime_index_t...i)
    {
      using RC = row_coefficient_types_of_t<Arg>;

      if constexpr (has_uniform_dimension_type<column_coefficient_types_of_t<Arg>>)
      {
        using CC = typename uniform_dimension_type_of_t<column_coefficient_types_of_t<Arg>>;
        return make_matrix<RC, CC>(column<index...>(to_covariance_nestable(std::forward<Arg>(arg)), i...));
      }
      else if constexpr (fixed_index_descriptor<column_coefficient_types_of_t<Arg>>)
      {
        static_assert(sizeof...(index) > 0);
        using CC = column_coefficient_types_of_t<Arg>::template Coefficient<index...>;
        static_assert(dimension_size_of_v<CC> == 1);
        return make_matrix<RC, CC>(column<index...>(to_covariance_nestable(std::forward<Arg>(arg))));
      }
      else
      {
        static_assert(dynamic_index_descriptor<column_coefficient_types_of_t<Arg>>);
        using CC = column_coefficient_types_of_t<Arg>::template Coefficient<index...>;
        return make_matrix<RC, CC>(column(to_covariance_nestable(std::forward<Arg>(arg)), i...));
      }
    }


    template<std::size_t...index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    row(Arg&& arg, runtime_index_t...i)
    {
      using CC = column_coefficient_types_of_t<Arg>;

      if constexpr (has_uniform_dimension_type<row_coefficient_types_of_t<Arg>>)
      {
        using RC = typename uniform_dimension_type_of_t<row_coefficient_types_of_t<Arg>>;
        return make_matrix<RC, CC>(column<index...>(to_covariance_nestable(std::forward<Arg>(arg)), i...));
      }
      else if constexpr (fixed_index_descriptor<row_coefficient_types_of_t<Arg>>)
      {
        static_assert(sizeof...(index) > 0);
        using RC = row_coefficient_types_of_t<Arg>::template Coefficient<index...>;
        static_assert(dimension_size_of_v<RC> == 1);
        return make_matrix<RC, CC>(column<index...>(to_covariance_nestable(std::forward<Arg>(arg))));
      }
      else
      {
        static_assert(dynamic_index_descriptor<row_coefficient_types_of_t<Arg>>);
        using RC = row_coefficient_types_of_t<Arg>::template Coefficient<index...>;
        return make_matrix<RC, CC>(column<index...>(to_covariance_nestable(std::forward<Arg>(arg)), i...));
      }
    }
  };


#ifdef __cpp_concepts
    template<covariance T>
    struct ElementAccess<T>
#else
    template<typename T>
    struct ElementAccess<T, std::enable_if_t<covariance<T>>>
#endif
    {
    };


#ifdef __cpp_concepts
    template<covariance T>
    struct ArrayOperations<T>
#else
    template<typename T>
    struct ArrayOperations<T, std::enable_if_t<covariance<T>>>
#endif
    {

      template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
      static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
      {
        return OpenKalman::fold<order>(b, std::forward<Accum>(accum), to_covariance_nestable(std::forward<Arg>(arg)));
      }

    };


#ifdef __cpp_concepts
  template<covariance T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<covariance<T>>>
#endif
  {

    template<typename Arg>
    static auto
    diagonal_of(Arg&& arg) noexcept
    {
      using C = row_coefficient_types_of_t<Arg>;
      auto b = make_self_contained<Arg>(diagonal_of(oin::to_covariance_nestable(std::forward<Arg>(arg))));
      return Matrix<C, Axis, decltype(b)>(std::move(b));
    }

  };


#ifdef __cpp_concepts
    template<covariance T>
    struct LinearAlgebra<T>
#else
    template<typename T>
    struct linearAlgebra<T, std::enable_if_t<covariance<T>>>
#endif
    {

      template<typename Arg>
      static constexpr decltype(auto) conjugate(Arg&& arg) noexcept
      {
        // \todo optimize this by also copying cholesky nested matrix
        return MatrixTraits<Arg>::make(OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr decltype(auto) transpose(Arg&& arg) noexcept
      {
        // \todo optimize this by also copying cholesky nested matrix
        return MatrixTraits<Arg>::make(OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr decltype(auto) adjoint(Arg&& arg) noexcept
      {
        // \todo optimize this by also copying cholesky nested matrix
        static_assert(triangular_covariance<Arg>)
        return MatrixTraits<Arg>::make(OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr auto determinant(Arg&& arg) noexcept
      {
        return std::forward<Arg>(arg).determinant();
      }


      template<typename Arg>
      static constexpr auto trace(Arg&& arg) noexcept
      {
        // \todo Optimize this?
        return OpenKalman::trace(to_covariance_nestable(std::forward<Arg>(arg)));
      }


      template<TriangleType t, typename Arg, typename U, typename Alpha>
      static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
      {
        if constexpr (std::is_same_v<A&&, std::decay_t<A>&>)
        {
          return arg.rank_update(std::forward<U>(u), alpha);
        }
        else
        {
          auto ret = std::forward<A>(a).rank_update(std::forward<U>(u), alpha);
          return ret;
        }
      }


      template<TriangleType t, typename Arg, typename U, typename Alpha>
      static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
      {
        if constexpr (std::is_same_v<A&&, std::decay_t<A>&>)
        {
          return arg.rank_update(std::forward<U>(u), alpha);
        }
        else
        {
          auto ret = std::forward<A>(a).rank_update(std::forward<U>(u), alpha);
          return ret;
        }
      }


      template<bool must_be_unique, bool must_be_exact, typename A, typename B>
      inline auto
      solve(A&& a, B&& b) noexcept
      {
        auto x = make_self_contained<A, B>(OpenKalman::solve<must_be_unique, must_be_exact>(
          to_covariance_nestable(std::forward<A>(a)), nested_matrix(std::forward<B>(b))));
        return MatrixTraits<B>::template make<row_coefficient_types_of_t<A>>(std::move(x));
      }

    };

  } // namespace interface


#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg)
  {
    using RC = row_coefficient_types_of_t<Arg>;
    return make_matrix<RC, Axis>(reduce_columns(to_covariance_nestable(std::forward<Arg>(arg))));
  }


  /// Perform an LQ decomposition of matrix A=[L,0]Q, where L is a lower-triangular matrix, and Q is orthogonal.
  /// Returns L as a lower-triangular matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  LQ_decomposition(Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    auto tm = LQ_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
    return make_square_root_covariance<C>(std::move(tm));
  }


  /// Perform a QR decomposition of matrix A=Q[U,0], where U is an upper-triangular matrix, and Q is orthogonal.
  /// Returns L as an upper-triangular matrix.
#ifdef __cpp_concepts
  template<covariance Arg>
#else
  template<typename Arg, std::enable_if_t<covariance<Arg>, int> = 0>
#endif
  inline auto
  QR_decomposition(Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    auto tm = QR_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
    return make_square_root_covariance<C>(std::move(tm));
  }


  /// Concatenate one or more Covariance or SquareRootCovariance objects diagonally.
#ifdef __cpp_concepts
  template<covariance M, covariance ... Ms>
#else
  template<typename M, typename ... Ms, std::enable_if_t<(covariance<M> and ... and covariance<Ms>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(M&& m, Ms&& ... mN) noexcept
  {
    if constexpr(sizeof...(Ms) > 0)
    {
      using Coeffs =
        Concatenate<row_coefficient_types_of_t<M>, row_coefficient_types_of_t<Ms>...>;
      auto cat = concatenate_diagonal(nested_matrix(std::forward<M>(m)), nested_matrix(std::forward<Ms>(mN))...);
      return MatrixTraits<M>::template make<Coeffs>(std::move(cat));
    }
    else
    {
      return std::forward<M>(m);
    }
  }


  namespace detail
  {
    template<typename C, typename Expr, typename Arg>
    inline auto
    split_item_impl(Arg&& arg)
    {
      if constexpr(one_by_one_matrix<Arg> and self_adjoint_covariance<Expr> and cholesky_form<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_by_one_matrix<Arg> and triangular_covariance<Expr> and not cholesky_form<Expr>)
      {
        return MatrixTraits<Expr>::template make<C>(Cholesky_factor<TriangleType::lower>(std::forward<Arg>(arg)));
      }
      else
      {
        return MatrixTraits<Expr>::template make<C>(std::forward<Arg>(arg));
      }
    }
  }

  namespace internal
  {
    template<typename Expr, typename F, typename Arg>
    inline auto split_cov_diag_impl(const F& f, Arg&& arg)
    {
      if constexpr(one_by_one_matrix<Arg> and self_adjoint_covariance<Expr> and cholesky_form<Expr>)
      {
        return f(Cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_by_one_matrix<Arg> and triangular_covariance<Expr> and not cholesky_form<Expr>)
      {
        return f(Cholesky_factor<TriangleType::lower>(std::forward<Arg>(arg)));
      }
      else
      {
        return f(std::forward<Arg>(arg));
      }
    }

    template<typename Expr>
    struct SplitCovDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        static_assert(equivalent_to<RC, CC>);
        auto f = [](auto&& m) { return MatrixTraits<Expr>::template make<RC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename CC>
    struct SplitCovVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return make_matrix<RC, CC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename RC>
    struct SplitCovHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return make_matrix<RC, CC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };
  }

  /// Split Covariance or SquareRootCovariance diagonally.
#ifdef __cpp_concepts
  template<typed_index_descriptor ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_diagonal(M&& m) noexcept
  {
    static_assert(prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<M>>);
    return split_diagonal<oin::SplitCovDiagF<M>, Cs...>(nested_matrix(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<typed_index_descriptor ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_vertical(M&& m) noexcept
  {
    using CC = row_coefficient_types_of_t<M>;
    static_assert(prefix_of<Concatenate<Cs...>, CC>);
    return split_vertical<oin::SplitCovVertF<M, CC>, Cs...>(make_dense_writable_matrix_from(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<typed_index_descriptor ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    using RC = row_coefficient_types_of_t<M>;
    static_assert(prefix_of<Concatenate<Cs...>, RC>);
    return split_horizontal<oin::SplitCovHorizF<M, RC>, Cs...>(make_dense_writable_matrix_from(std::forward<M>(m)));
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires
    requires(Arg&& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      column_dimension_of_v<decltype(f(column<0>(arg)))> == 1;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f] (auto&& col) -> auto {
      return make_self_contained(nested_matrix(f(make_matrix<C, Axis>(std::forward<decltype(col)>(col)))));
    };
    return make_matrix<C, C>(apply_columnwise(f_nested, to_covariance_nestable(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires
    requires(Arg&& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      column_dimension_of_v<decltype(f(column<0>(arg), i))> == 1;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f] (auto&& col, std::size_t i) -> auto {
      return make_self_contained(nested_matrix(f(make_matrix<C, Axis>(std::forward<decltype(col)>(col)), i)));
    };
    return make_matrix<C, C>(apply_columnwise(f_nested, to_covariance_nestable(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires std::convertible_to<
    std::invoke_result_t<Function, scalar_type_of_t<Arg>>, const scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<covariance<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename scalar_type_of<Arg>::type>,
      const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    return make_matrix<C, C>(apply_coefficientwise(f, to_covariance_nestable(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires std::convertible_to<
    std::invoke_result_t<Function, scalar_type_of_t<Arg>, std::size_t, std::size_t>,
      const scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<covariance<Arg> and
    std::is_convertible_v<std::invoke_result_t<Function, typename scalar_type_of<Arg>::type, std::size_t, std::size_t>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    using C = row_coefficient_types_of_t<Arg>;
    return make_matrix<C, C>(apply_coefficientwise(f, to_covariance_nestable(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<covariance Cov>
#else
  template<typename Cov, std::enable_if_t<covariance<Cov>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const Cov& c)
  {
    os << make_dense_writable_matrix_from(c);
    return os;
  }


}

#endif //OPENKALMAN_COVARIANCEOVERLOADS_HPP
