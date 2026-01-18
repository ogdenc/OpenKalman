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
#ifdef __cpp_concepts
  template<self_adjoint_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<self_adjoint_covariance<Arg>, int> = 0>
#endif
  inline auto
  square_root(Arg&& arg)
  {
    return std::forward<Arg>(arg).square_root();
  }


#ifdef __cpp_concepts
  template<triangular_covariance Arg>
#else
  template<typename Arg, std::enable_if_t<triangular_covariance<Arg>, int> = 0>
#endif
  inline auto
  square(Arg&& arg)
  {
    return std::forward<Arg>(arg).square();
  }


  namespace interface
  {

#ifdef __cpp_concepts
    template<covariance T>
    struct library_interface<T>
#else
    template<typename T>
    struct linearAlgebra<T, std::enable_if_t<covariance<T>>>
#endif
    {
      template<typename Derived>
      using library_base = internal::library_base_t<Derived, nested_object_of_t<T>>;


      template<data_layout layout, typename Scalar, typename...D>
      static auto
      make_default(D&&...d)
      {
        return library_interface<nested_object_of_t<T>>::template make_default<layout, Scalar>(std::forward<D>(d)...);
      }


      // fill_components not necessary because T is not a dense writable matrix.


      template<typename Arg>
      static decltype(auto)
      to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<nested_object_of_t<T>>(
          OpenKalman::internal::to_covariance_nestable(std::forward<Arg>(arg))(std::forward<Arg>(arg)));
      }

      template<typename C, typenameD>
      static constexpr auto
      make_constant(C&& c, D&& d)
      {
        return make_constant<nested_object_of_t<T>>(std::forward<C>(c), std::forward<D>(d));
      }


      template<typename Scalar, typename...D>
      static constexpr auto
      make_identity_matrix(D&&...d)
      {
        return make_identity_matrix_like<nested_object_of_t<T>, Scalar>(std::forward<D>(d)...);
      }


      template<typename Arg, typename...Begin, typename...Size>
      static decltype(auto)
      get_slice(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
      {
        /// \todo Properly wrap this
        return OpenKalman::get_slice(nested_object(std::forward<Arg>(arg)), begin, size);
      };


      template<typename Arg, typename Block, typename...Begin>
      static Arg&
      set_slice(Arg& arg, Block&& block, Begin...begin)
      {
        /// \todo Properly wrap this
        return OpenKalman::set_slice(nested_object(std::forward<Arg>(arg)), std::forward<Block>(block), begin...);
      };


      template<triangle_type t, typename A, typename B>
      static decltype(auto)
      set_triangle(A&& a, B&& b)
      {
        /// \todo Properly wrap this
        return OpenKalman::internal::set_triangle<t>(nested_object(std::forward<A>(a)), std::forward<B>(b));
      }


      template<typename Arg>
      static auto
      diagonal_of(Arg&& arg)
      {
        using C = vector_space_descriptor_of_t<Arg, 0>;
        auto b = make_self_contained<Arg>(diagonal_of(oin::to_covariance_nestable(std::forward<Arg>(arg))));
        return Matrix<C, Axis, decltype(b)>(std::move(b));
      }


      template<typename Arg>
      static constexpr decltype(auto) conjugate(Arg&& arg)
      {
        // \todo optimize this by also copying cholesky nested matrix
        return MatrixTraits<std::decay_t<Arg>>::make(OpenKalman::conjugate(nested_object(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr decltype(auto) transpose(Arg&& arg)
      {
        // \todo optimize this by also copying cholesky nested matrix
        return MatrixTraits<std::decay_t<Arg>>::make(OpenKalman::transpose(nested_object(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr decltype(auto) conjugate_transpose(Arg&& arg)
      {
        // \todo optimize this by also copying cholesky nested matrix
        static_assert(triangular_covariance<Arg>);
        return MatrixTraits<std::decay_t<Arg>>::make(OpenKalman::conjugate_transpose(nested_object(std::forward<Arg>(arg))));
      }


      template<typename Arg>
      static constexpr auto determinant(Arg&& arg)
      {
        return std::forward<Arg>(arg).determinant();
      }


      template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
      {
        if constexpr (std::is_same_v<A&&, std::decay_t<A>&>)
        {
          return a.rank_update(std::forward<U>(u), alpha);
        }
        else
        {
          auto ret {std::forward<A>(a).rank_update(std::forward<U>(u), alpha)};
          return ret;
        }
      }


      template<triangle_type triangle, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
      {
        if constexpr (std::is_same_v<A&&, std::decay_t<A>&>)
        {
          return a.rank_update(std::forward<U>(u), alpha);
        }
        else
        {
          auto ret {std::forward<A>(a).rank_update(std::forward<U>(u), alpha)};
          return ret;
        }
      }


      template<bool must_be_unique, bool must_be_exact, typename A, typename B>
      inline auto
      solve(A&& a, B&& b)
      {
        auto x = make_self_contained<A, B>(OpenKalman::solve<must_be_unique, must_be_exact>(
          to_covariance_nestable(std::forward<A>(a)), nested_object(std::forward<B>(b))));
        return MatrixTraits<std::decay_t<B>>::template make<vector_space_descriptor_of_t<A, 0>>(std::move(x));
      }


      template<typename Arg>
      inline auto
      LQ_decomposition(Arg&& arg)
      {
        return LQ_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
      }


      template<typename Arg>
      inline auto
      QR_decomposition(Arg&& arg)
      {
        return QR_decomposition(to_covariance_nestable(std::forward<Arg>(arg)));
      }

    };

  }


  /// Concatenate one or more Covariance or SquareRootCovariance objects diagonally.
#ifdef __cpp_concepts
  template<covariance M, covariance ... Ms>
#else
  template<typename M, typename ... Ms, std::enable_if_t<(covariance<M> and ... and covariance<Ms>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(M&& m, Ms&& ... mN)
  {
    if constexpr(sizeof...(Ms) > 0)
    {
      using Coeffs =
        static_concatenate_t<vector_space_descriptor_of_t<M, 0>, vector_space_descriptor_of_t<Ms, 0>...>;
      auto cat = concatenate_diagonal(nested_object(std::forward<M>(m)), nested_object(std::forward<Ms>(mN))...);
      return MatrixTraits<std::decay_t<M>>::template make<Coeffs>(std::move(cat));
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
      if constexpr(one_dimensional<Arg> and self_adjoint_covariance<Expr> and cholesky_form<Expr>)
      {
        return MatrixTraits<std::decay_t<Expr>>::template make<C>(cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_dimensional<Arg> and triangular_covariance<Expr> and not cholesky_form<Expr>)
      {
        return MatrixTraits<std::decay_t<Expr>>::template make<C>(cholesky_factor<triangle_type::lower>(std::forward<Arg>(arg)));
      }
      else
      {
        return MatrixTraits<std::decay_t<Expr>>::template make<C>(std::forward<Arg>(arg));
      }
    }
  }

  namespace internal
  {
    template<typename Expr, typename F, typename Arg>
    inline auto split_cov_diag_impl(const F& f, Arg&& arg)
    {
      if constexpr(one_dimensional<Arg> and self_adjoint_covariance<Expr> and cholesky_form<Expr>)
      {
        return f(cholesky_square(std::forward<Arg>(arg)));
      }
      else if constexpr(one_dimensional<Arg> and triangular_covariance<Expr> and not cholesky_form<Expr>)
      {
        return f(cholesky_factor<triangle_type::lower>(std::forward<Arg>(arg)));
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
        static_assert(compares_with<RC, CC>);
        auto f = [](auto&& m) { return MatrixTraits<std::decay_t<Expr>>::template make<RC>(std::forward<decltype(m)>(m)); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename CC>
    struct SplitCovVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return attach_patterns(std::forward<decltype(m)>(m), RC{}, CC{}); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };

    template<typename Expr, typename RC>
    struct SplitCovHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        auto f = [](auto&& m) { return attach_patterns(std::forward<decltype(m)>(m), RC{}, CC{}); };
        return split_cov_diag_impl<Expr>(f, std::forward<Arg>(arg));
      }
    };
  }

  /// Split Covariance or SquareRootCovariance diagonally.
#ifdef __cpp_concepts
  template<fixed_pattern ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_diagonal(M&& m)
  {
    static_assert(patterns::compares_with<static_concatenate_t<Cs...>, vector_space_descriptor_of_t<M, 0>, less_equal<>>);
    return split_diagonal<oin::SplitCovDiagF<M>, Cs...>(nested_object(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<fixed_pattern ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_vertical(M&& m)
  {
    using CC = vector_space_descriptor_of_t<M, 0>;
    static_assert(patterns::compares_with<static_concatenate_t<Cs...>, CC, less_equal<>>);
    return split_vertical<oin::SplitCovVertF<M, CC>, Cs...>(to_dense_object(std::forward<M>(m)));
  }


  /// Split Covariance or SquareRootCovariance vertically. Result is a tuple of typed matrices.
#ifdef __cpp_concepts
  template<fixed_pattern ... Cs, covariance M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<covariance<M>, int> = 0>
#endif
  inline auto
  split_horizontal(M&& m)
  {
    using RC = vector_space_descriptor_of_t<M, 0>;
    static_assert(patterns::compares_with<static_concatenate_t<Cs...>, RC, less_equal<>>);
    return split_horizontal<oin::SplitCovHorizF<M, RC>, Cs...>(to_dense_object(std::forward<M>(m)));
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires
    requires(Arg&& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      index_dimension_of_v<decltype(f(column<0>(arg))), 1> == 1;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    const auto f_nested = [&f] (auto&& col) -> auto {
      return make_self_contained(nested_object(f(attach_patterns(std::forward<decltype(col)>(col), C{}, Axis{}))));
    };
    return attach_patterns(apply_columnwise(f_nested, to_covariance_nestable(std::forward<Arg>(arg))), C{}, C{});
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires
    requires(Arg&& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      index_dimension_of_v<decltype(f(column<0>(arg), i)), 1> == 1;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<
    covariance<Arg> and typed_matrix<std::invoke_result_t<
      Function, std::decay_t<decltype(column<0>(std::declval<Arg>()))>, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    const auto f_nested = [&f] (auto&& col, std::size_t i) -> auto {
      return make_self_contained(nested_object(f(attach_patterns(std::forward<decltype(col)>(col), C{}, Axis{}), i)));
    };
    return attach_patterns(apply_columnwise(f_nested, to_covariance_nestable(std::forward<Arg>(arg))), C{}, C{});
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires std::convertible_to<
    std::invoke_result_t<Function, scalar_type_of_t<Arg>>, const scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<covariance<Arg> and
    stdex::convertible_to<std::invoke_result_t<Function, typename scalar_type_of<Arg>::type>,
      const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    return attach_patterns(apply_coefficientwise(f, to_covariance_nestable(std::forward<Arg>(arg))), C{}, C{});
  }


#ifdef __cpp_concepts
  template<typename Function, covariance Arg> requires std::convertible_to<
    std::invoke_result_t<Function, scalar_type_of_t<Arg>, std::size_t, std::size_t>,
      const scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<covariance<Arg> and
    stdex::convertible_to<std::invoke_result_t<Function, typename scalar_type_of<Arg>::type, std::size_t, std::size_t>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    using C = vector_space_descriptor_of_t<Arg, 0>;
    return attach_patterns(apply_coefficientwise(f, to_covariance_nestable(std::forward<Arg>(arg))), C{}, C{});
  }


#ifdef __cpp_concepts
  template<covariance Cov>
#else
  template<typename Cov, std::enable_if_t<covariance<Cov>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const Cov& c)
  {
    os << to_dense_object(c);
    return os;
  }


}

#endif
