/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_TYPED_MATRIX_OVERLOADS_HPP
#define OPENKALMAN_TYPED_MATRIX_OVERLOADS_HPP

#include <iostream>

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<typed_matrix T>
  struct library_interface<T>
#else
  template<typename T>
  struct linearAlgebra<T, std::enable_if_t<typed_matrix<T>>>
#endif
  {
    template<typename Derived>
    using LibraryBase = internal::library_base_t<Derived, nested_object_of_t<T>>;


    template<typename Arg>
    static decltype(auto) to_native_matrix(Arg&& arg)
    {
      return OpenKalman::to_native_matrix<nested_object_of_t<T>>(nested_object(std::forward<Arg>(arg)));
    }


    template<Layout layout, typename Scalar, typename...D>
    static auto make_default(D&&...d)
    {
      return library_interface<nested_object_of_t<T>>::template make_default<layout, Scalar>(std::forward<D>(d)...);
    }


    // fill_components not necessary because T is not a dense writable matrix.


    template<typename C, typename...D>
    static constexpr auto make_constant(C&& c, D&&...d)
    {
      return make_constant<nested_object_of_t<T>>(std::forward<C>(c), std::forward<D>(d)...);
    }


    template<typename Scalar, typename...D>
    static constexpr auto make_identity_matrix(D&&...d)
    {
      return make_identity_matrix_like<nested_object_of_t<T>, Scalar>(std::forward<D>(d)...);
    }


    template<typename Arg, typename...Begin, typename...Size>
    static decltype(auto) get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      /// \todo Properly wrap this
      return OpenKalman::get_block(nested_object(std::forward<Arg>(arg)), begin, size);
    };


    template<typename Arg, typename Block, typename...Begin>
    static Arg& set_block(Arg& arg, Block&& block, Begin...begin)
    {
      /// \todo Properly wrap this
      return OpenKalman::set_block(nested_object(std::forward<Arg>(arg)), std::forward<Block>(block), begin...);
    };


    template<TriangleType t, typename A, typename B>
    static decltype(auto) set_triangle(A&& a, B&& b)
    {
      /// \todo Properly wrap this
      return OpenKalman::internal::set_triangle<t>(nested_object(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename Arg>
    static auto
    to_diagonal(Arg&& arg) noexcept
    {
      using C = vector_space_descriptor_of_t<Arg, 0>;
      auto b = to_diagonal(nested_object(std::forward<Arg>(arg)));
      return Matrix<C, C, decltype(b)>(std::move(b));
    }


    template<typename Arg>
    static auto
    diagonal_of(Arg&& arg) noexcept
    {
      using C = vector_space_descriptor_of_t<Arg, 0>;
      auto b = diagonal_of(nested_object(std::forward<Arg>(arg)));
      return Matrix<C, Axis, decltype(b)>(std::move(b));
    }


#ifdef __cpp_concepts
    template<mean Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<mean<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, const C& c) noexcept
    {
      auto&& n = OpenKalman::to_euclidean(nested_object(std::forward<Arg>(arg), std::forward<DC>(dc)...), c);
      return make_euclidean_mean<C>(std::forward<decltype(n)>(n), c);
    }


#ifdef __cpp_concepts
    template<euclidean_mean Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<euclidean_mean<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, const C& c) noexcept
    {
      auto&& n = OpenKalman::from_euclidean(nested_object(std::forward<Arg>(arg), c));
      return make_mean<C>(std::forward<decltype(n)>(n), c);
    }


#ifdef __cpp_concepts
    template<mean Arg, vector_space_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<mean<Arg> and vector_space_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C& c) noexcept
    {
      auto&& n = OpenKalman::wrap_angles(nested_object(std::forward<Arg>(arg), c));
      return MatrixTraits<std::decay_t<Arg>>::make(std::forward<decltype(n)>(n), c);
    }

    template<typename Arg>
    static constexpr auto conjugate(Arg&& arg) noexcept
    {
      using CRows = vector_space_descriptor_of_t<Arg, 0>;
      using CCols = vector_space_descriptor_of_t<Arg, 1>;
      if constexpr(euclidean_transformed<Arg>)
      {
        auto b = OpenKalman::conjugate(nested_object(from_euclidean(std::forward<Arg>(arg))));
        return Matrix<CRows, CCols, decltype(b)>(std::move(b));
      }
      else
      {
        auto b = OpenKalman::conjugate(nested_object(std::forward<Arg>(arg)));
        return Matrix<CRows, CCols, decltype(b)>(std::move(b));
      }
    }


    template<typename Arg>
    static constexpr auto transpose(Arg&& arg) noexcept
    {
      using CRows = vector_space_descriptor_of_t<Arg, 0>;
      using CCols = vector_space_descriptor_of_t<Arg, 1>;
      if constexpr(euclidean_transformed<Arg>)
      {
        auto b = OpenKalman::transpose(nested_object(from_euclidean(std::forward<Arg>(arg))));
        return Matrix<CCols, CRows, decltype(b)>(std::move(b));
      }
      else
      {
        auto b = OpenKalman::transpose(nested_object(std::forward<Arg>(arg)));
        return Matrix<CCols, CRows, decltype(b)>(std::move(b));
      }
    }


    template<typename Arg>
    static constexpr auto adjoint(Arg&& arg) noexcept
    {
      using CRows = vector_space_descriptor_of_t<Arg, 0>;
      using CCols = vector_space_descriptor_of_t<Arg, 1>;
      if constexpr(euclidean_transformed<Arg>)
      {
        auto b = OpenKalman::adjoint(nested_object(from_euclidean(std::forward<Arg>(arg))));
        return Matrix<CCols, CRows, decltype(b)>(std::move(b));
      }
      else
      {
        auto b = OpenKalman::adjoint(nested_object(std::forward<Arg>(arg)));
        return Matrix<CCols, CRows, decltype(b)>(std::move(b));
      }
    }


    template<typename Arg>
    static constexpr auto determinant(Arg&& arg) noexcept
    {
      return OpenKalman::determinant(nested_object(std::forward<Arg>(arg)));
    }


    template<typename A, typename B>
    static decltype(auto) contract(A&& a, B&& b)
    {
      return OpenKalman::contract(nested_object(std::forward<A>(a)), nested_object(std::forward<B>(b)));
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_hermitian(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_hermitian(nested_object(std::forward<A>(a)), nested_object(std::forward<U>(u)), alpha);
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_triangular(
        nested_object(std::forward<A>(a)), nested_object(std::forward<U>(u)), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    inline auto
    solve(A&& a, B&& b) noexcept
    {
      auto x = OpenKalman::solve<must_be_unique, must_be_exact>(
        nested_object(std::forward<A>(a)), nested_object(std::forward<B>(b)));
      return MatrixTraits<std::decay_t<B>>::make(std::move(x));
    }


    template<typename A>
    static inline auto
    LQ_decomposition(A&& a)
    {
      return LQ_decomposition(nested_object(std::forward<A>(a)));
    }


    template<typename A>
    static inline auto
    QR_decomposition(A&& a)
    {
      return QR_decomposition(nested_object(std::forward<A>(a)));
    }

  };

} // namespace OpenKalman::interface


namespace OpenKalman
{

  /// Concatenate one or more typed matrices objects vertically.
#ifdef __cpp_concepts
  template<typed_matrix V, typed_matrix ... Vs> requires (sizeof...(Vs) == 0) or
    (equivalent_to<vector_space_descriptor_of_t<V, 1>, vector_space_descriptor_of_t<Vs, 1>> and ...)
#else
  template<typename V, typename ... Vs, std::enable_if_t<(typed_matrix<V> and ... and typed_matrix<Vs>) and
    ((sizeof...(Vs) == 0) or (equivalent_to<vector_space_descriptor_of_t<V, 1>,
      vector_space_descriptor_of_t<Vs, 1>> and ...)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = concatenate_fixed_vector_space_descriptor_t<vector_space_descriptor_of_t<V, 0>, vector_space_descriptor_of_t<Vs, 0>...>;
      return MatrixTraits<std::decay_t<V>>::template make<RC>(
        concatenate_vertical(nested_object(std::forward<V>(v)), nested_object(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate one or more typed matrices vertically. (Synonym for concatenate_vertical.)
#ifdef __cpp_concepts
  template<typed_matrix ... Vs>
#else
  template<typename ... Vs, std::enable_if_t<(typed_matrix<Vs> and ...), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate(Vs&& ... vs) noexcept
  {
    return concatenate_vertical(std::forward<Vs>(vs)...);
  };


  /// Concatenate one or more matrix objects vertically.
#ifdef __cpp_concepts
template<typed_matrix V, typed_matrix ... Vs> requires (sizeof...(Vs) == 0) or
    (equivalent_to<vector_space_descriptor_of_t<V, 0>, vector_space_descriptor_of_t<Vs, 0>> and ...)
#else
template<typename V, typename ... Vs, std::enable_if_t<(typed_matrix<V> and ... and typed_matrix<Vs>) and
    ((sizeof...(Vs) == 0) or (equivalent_to<vector_space_descriptor_of_t<V, 0>,
      vector_space_descriptor_of_t<Vs, 0>> and ...)), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = vector_space_descriptor_of_t<V, 0>;
      using CC = concatenate_fixed_vector_space_descriptor_t<vector_space_descriptor_of_t<V, 1>,
      vector_space_descriptor_of_t<Vs, 1>...>;
      auto cat = concatenate_horizontal(nested_object(std::forward<V>(v)), nested_object(std::forward<Vs>(vs))...);
      if constexpr(euclidean_vector_space_descriptor<CC>)
      {
        return MatrixTraits<std::decay_t<V>>::template make<RC, CC>(std::move(cat));
      }
      else
      {
        return make_matrix<RC, CC>(std::move(cat));
      }
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  /// Concatenate one or more typed matrices diagonally.
#ifdef __cpp_concepts
  template<typed_matrix V, typed_matrix ... Vs>
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (typed_matrix<V> and ... and typed_matrix<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_diagonal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr(sizeof...(Vs) > 0)
    {
      using RC = concatenate_fixed_vector_space_descriptor_t<vector_space_descriptor_of_t<V, 0>, vector_space_descriptor_of_t<Vs, 0>...>;
      using CC = concatenate_fixed_vector_space_descriptor_t<vector_space_descriptor_of_t<V, 1>,
        vector_space_descriptor_of_t<Vs, 1>...>;
      return MatrixTraits<std::decay_t<V>>::template make<RC, CC>(
        concatenate_diagonal(nested_object(std::forward<V>(v)), nested_object(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  };


  namespace internal
  {
    template<typename Expr, typename CC>
    struct SplitMatVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        return MatrixTraits<std::decay_t<Expr>>::template make<RC, CC>(std::forward<Arg>(arg));
      }
    };


    template<typename Expr, typename RC>
    struct SplitMatHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return MatrixTraits<std::decay_t<Expr>>::template make<RC, CC>(std::forward<Arg>(arg));
      }
    };


    template<typename Expr>
    struct SplitMatDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        static_assert(equivalent_to<RC, CC>);
        return MatrixTraits<std::decay_t<Expr>>::template make<RC, CC>(std::forward<Arg>(arg));
      }
    };
  }


  /// Split typed matrix into one or more typed matrices vertically.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor ... Cs, typed_matrix M> requires
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, vector_space_descriptor_of_t<M, 0>>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M> and
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, vector_space_descriptor_of_t<M, 0>>, int> = 0>
#endif
  inline auto
  split_vertical(M&& m) noexcept
  {
    using CC = vector_space_descriptor_of_t<M, 1>;
    constexpr auto euclidean = euclidean_transformed<M>;
    return split_vertical<oin::SplitMatVertF<M, CC>, euclidean, Cs...>(nested_object(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor ... Cs, typed_matrix M> requires
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, vector_space_descriptor_of_t<M, 1>>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M> and
    internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, vector_space_descriptor_of_t<M, 1>>, int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    using RC = vector_space_descriptor_of_t<M, 0>;
    return split_horizontal<oin::SplitMatHorizF<M, RC>, Cs...>(nested_object(std::forward<M>(m)));
  }


  /// Split typed matrix into one or more typed matrices horizontally. Column coefficients must all be Axis.
#ifdef __cpp_concepts
  template<std::size_t ... cuts, typed_matrix M> requires has_untyped_index<M, 1> and (sizeof...(cuts) > 0) and
    ((... + cuts) <= index_dimension_of_v<M, 1>)
#else
  template<std::size_t ... cuts, typename M,
    std::enable_if_t<typed_matrix<M> and has_untyped_index<M, 1> and (sizeof...(cuts) > 0) and
      ((0 + ... + cuts) <= index_dimension_of<M, 1>::value), int> = 0>
#endif
  inline auto
  split_horizontal(M&& m) noexcept
  {
    return split_horizontal<Dimensions<cuts>...>(std::forward<M>(m));
  }


  /// Split typed matrix into one or more typed matrices diagonally.
#ifdef __cpp_concepts
  template<fixed_vector_space_descriptor ... Cs, typed_matrix M>
#else
  template<typename ... Cs, typename M, std::enable_if_t<typed_matrix<M>, int> = 0>
#endif
  inline auto
  split_diagonal(M&& m) noexcept
  {
    static_assert(internal::prefix_of<concatenate_fixed_vector_space_descriptor_t<Cs...>, vector_space_descriptor_of_t<M, 0>>);
    static_assert(equivalent_to<vector_space_descriptor_of_t<M, 0>::ColumnCoefficients, MatrixTraits<std::decay_t<M>>>);
    return split_diagonal<oin::SplitMatDiagF<M>, Cs...>(nested_object(std::forward<M>(m)));
  }


  ////

#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires has_untyped_index<Arg, 1> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); } and
    (not std::is_const_v<std::remove_reference_t<nested_object_of_t<Arg>>>) and
    modifiable<nested_object_of_t<Arg>, nested_object_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1> and
    std::is_invocable_v<const Function&,
      std::decay_t<decltype(column(std::declval<Arg&>(), 0))>& > and
    (not std::is_const_v<std::remove_reference_t<nested_object_of_t<Arg>>>) and
    modifiable<nested_object_of_t<Arg>, nested_object_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using RC = vector_space_descriptor_of_t<Arg, 0>;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<std::decay_t<Arg>>::template make<RC, Axis>(std::move(col));
      f(mc);
      col = std::move(nested_object(mc));
    };
    auto& c = nested_object(arg);
    apply_columnwise(f_nested, c);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires has_untyped_index<Arg, 1> and
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    } and
    (not std::is_const_v<std::remove_reference_t<nested_object_of_t<Arg>>>) and
    modifiable<nested_object_of_t<Arg>, nested_object_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1> and
    std::is_invocable_v<Function,
    std::decay_t<decltype(column(std::declval<Arg&>(), 0))>&, std::size_t> and
    (not std::is_const_v<std::remove_reference_t<nested_object_of_t<Arg>>>) and
    modifiable<nested_object_of_t<Arg>, nested_object_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using RC = vector_space_descriptor_of_t<Arg, 0>;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<std::decay_t<Arg>>::template make<RC, Axis>(std::move(col));
      f(mc, i);
      col = std::move(nested_object(mc));
    };
    auto& c = nested_object(arg);
    apply_columnwise(f_nested, c);
    if constexpr(wrapped_mean<Arg>) c = wrap_angles<RC>(c);
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires has_untyped_index<Arg, 1> and
    requires(const Arg& arg, const Function& f) {
      {f(column<0>(arg))} -> typed_matrix;
      {f(column<0>(arg))} -> vector;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&& >>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0))>;
    using ResRC = vector_space_descriptor_of_t<ResultType, 0>;
    using ResCC0 = vector_space_descriptor_of_t<ResultType, 1>;
    static_assert(dimension_size_of_v<ResCC0> == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = replicate_fixed_vector_space_descriptor_t<ResCC0, index_dimension_of_v<Arg, 1>>;
    using RC = vector_space_descriptor_of_t<Arg, 0>;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(
        nested_object(f(MatrixTraits<std::decay_t<Arg>>::template make<RC, Axis>(std::forward<decltype(col)>(col)))));
    };
    return MatrixTraits<std::decay_t<ResultType>>::template make<ResRC, ResCC>(apply_columnwise(f_nested, nested_object(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires has_untyped_index<Arg, 1> and
    requires(const Arg& arg, const Function& f, std::size_t i) {
      {f(column<0>(arg), i)} -> typed_matrix;
      {f(column<0>(arg), i)} -> vector;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and has_untyped_index<Arg, 1> and
    typed_matrix<std::invoke_result_t<const Function&,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    // \todo Make it so this function can accept any typed matrix with identically-typed columns.
    using ResultType = std::invoke_result_t<Function, decltype(column(std::declval<Arg&>(), 0)), std::size_t>;
    using ResRC = vector_space_descriptor_of_t<ResultType, 0>;
    using ResCC0 = vector_space_descriptor_of_t<ResultType, 1>;
    static_assert(dimension_size_of_v<ResCC0> == 1, "Function argument of apply_columnwise must return a column vector.");
    using ResCC = replicate_fixed_vector_space_descriptor_t<ResCC0, index_dimension_of_v<Arg, 1>>;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      using RC = vector_space_descriptor_of_t<Arg, 0>;
      return make_self_contained(
        nested_object(f(MatrixTraits<std::decay_t<Arg>>::template make<RC, Axis>(std::forward<decltype(col)>(col)), i)));
    };
    return MatrixTraits<std::decay_t<ResultType>>::template make<ResRC, ResCC>(apply_columnwise(f_nested, nested_object(arg)));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) {
      {f()} -> typed_matrix;
      {nested_object(f())} -> vector;
    }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    typed_matrix<std::invoke_result_t<const Function&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_nested = [&f] { return nested_object(f()); };
    using ResultType = std::invoke_result_t<Function>;
    using RC = vector_space_descriptor_of_t<ResultType, 0>;
    using CC0 = vector_space_descriptor_of_t<ResultType, 1>;
    static_assert(dimension_size_of_v<CC0> == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = replicate_fixed_vector_space_descriptor_t<CC0, count>;
    return MatrixTraits<std::decay_t<ResultType>>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) {
      {f(i)} -> typed_matrix;
      {nested_object(f(i))} -> vector;
    }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    typed_matrix<std::invoke_result_t<const Function&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    const auto f_nested = [&f](std::size_t i) { return nested_object(f(i)); };
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    using RC = vector_space_descriptor_of_t<ResultType, 0>;
    using CC0 = vector_space_descriptor_of_t<ResultType, 1>;
    static_assert(dimension_size_of_v<CC0> == 1, "Function argument of apply_columnwise must return a column vector.");
    using CC = replicate_fixed_vector_space_descriptor_t<CC0, count>;
    return MatrixTraits<std::decay_t<ResultType>>::template make<RC, CC>(apply_columnwise<count>(f_nested));
  }


  ////

#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>&>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>&>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(const Function& f, Arg& arg)
  {
    apply_coefficientwise(f, nested_object(arg));
    using RC = vector_space_descriptor_of_t<Arg, 0>;
    if constexpr(wrapped_mean<Arg>)
      nested_object(arg) = wrap_angles<RC>(nested_object(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires std::is_void_v<std::invoke_result_t<const Function&,
    std::decay_t<scalar_type_of_t<Arg>>&, std::size_t, std::size_t>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_void_v<std::invoke_result_t<const Function&,
      std::decay_t<scalar_type_of_t<Arg>>&, std::size_t, std::size_t>>, int> = 0>
#endif
  inline Arg&
  apply_coefficientwise(const Function& f, Arg& arg)
  {
    apply_coefficientwise(f, nested_object(arg));
    using RC = vector_space_descriptor_of_t<Arg, 0>;
    if constexpr(wrapped_mean<Arg>)
      nested_object(arg) = wrap_angles<RC>(nested_object(arg));
    return arg;
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires
    std::convertible_to<std::invoke_result_t<const Function&, std::decay_t<scalar_type_of_t<Arg>>>,
      scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&, std::decay_t<typename scalar_type_of<Arg>::type>>,
      typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, const Arg& arg)
  {
    return MatrixTraits<std::decay_t<Arg>>::make(apply_coefficientwise(f, nested_object(arg)));
  }


#ifdef __cpp_concepts
  template<typename Function, typed_matrix Arg> requires std::convertible_to<std::invoke_result_t<const Function&,
      std::decay_t<scalar_type_of_t<Arg>>, std::size_t, std::size_t>, scalar_type_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<typed_matrix<Arg> and
    std::is_convertible_v<std::invoke_result_t<const Function&,
      std::decay_t<typename scalar_type_of<Arg>::type>, std::size_t, std::size_t>,
      typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, const Arg& arg)
  {
    return MatrixTraits<std::decay_t<Arg>>::make(apply_coefficientwise(f, nested_object(arg)));
  }


#ifdef __cpp_concepts
  template<typed_matrix V, typename Function> requires
    std::convertible_to<std::invoke_result_t<const Function&>, scalar_type_of_t<V>>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and
    std::is_convertible_v<std::invoke_result_t<const Function&>, typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f)
  {
    constexpr auto rows = index_dimension_of_v<V, 0>;
    constexpr auto columns = index_dimension_of_v<V, 1>;
    using Scalar = scalar_type_of_t<V>;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function>>,
      std::decay_t<scalar_type_of_t<V>>>)
    {
      return MatrixTraits<std::decay_t<V>>::make(apply_coefficientwise<rows, columns>(f));
    }
    else
    {
      const auto f_conv = [&f] { return static_cast<Scalar>(f()); };
      return MatrixTraits<std::decay_t<V>>::make(apply_coefficientwise<rows, columns>(f_conv));
    }
  }


#ifdef __cpp_concepts
  template<typed_matrix V, typename Function> requires std::convertible_to<
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, scalar_type_of_t<V>>
#else
  template<typename V, typename Function, std::enable_if_t<typed_matrix<V> and std::is_convertible_v<
    std::invoke_result_t<const Function&, std::size_t, std::size_t>, typename scalar_type_of<V>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(Function&& f)
  {
    constexpr auto rows = index_dimension_of_v<V, 0>;
    constexpr auto columns = index_dimension_of_v<V, 1>;
    using Scalar = scalar_type_of_t<V>;
    if constexpr(std::is_same_v<
      std::decay_t<std::invoke_result_t<Function, std::size_t, std::size_t>>,
      std::decay_t<scalar_type_of_t<V>>>)
    {
      return MatrixTraits<std::decay_t<V>>::make(apply_coefficientwise<rows, columns>(std::forward<Function>(f)));
    }
    else
    {
      if constexpr (std::is_lvalue_reference_v<Function>)
      {
        auto f_conv = [&f](size_t i, size_t j){ return static_cast<Scalar>(f(i, j)); };
        return MatrixTraits<std::decay_t<V>>::make(apply_coefficientwise<rows, columns>(std::move(f_conv)));
      }
      else
      {
        auto f_conv = [f = std::move(f)](size_t i, size_t j){ return static_cast<Scalar>(f(i, j)); };
        return MatrixTraits<std::decay_t<V>>::make(apply_coefficientwise<rows, columns>(std::move(f_conv)));
      }
    }
  }


  /**
   * \brief Fill a fixed-shape typed matrix with random values selected from a random distribution.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *
   *  - One distribution for the entire matrix. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Matrix<Dimensions<2>, Dimensions<2>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}));
   *   \endcode
   *
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix n containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     auto n = randomize<Matrix<Dimensions<2>, Dimensions<2>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *
   *  - One distribution for each row. The following code constructs a 3-by-2 (o) or 2-by-2 (p) matrices
   *  in which elements in each row are selected according to the three (o) or two (p) listed distribution
   *  parameters:
   *   \code
   *     auto o = randomize<Matrix<Dimensions<3>, FixedDescriptor<angle::Radians, angle::Radians>, Eigen::Matrix<double, 3, 2>>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto p = randomize<Matrix<Dimensions<2>, FixedDescriptor<angle::Radians, angle::Radians>, Eigen::Matrix<double, 2, 2>>>(N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of p, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *
   *  - One distribution for each column. The following code constructs 2-by-3 matrix m
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto m = randomize<Matrix<FixedDescriptor<angle::Radians, angle::Radians>, Dimensions<3>, Eigen::Matrix<double, 2, 3>>>(N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   *
   * \tparam ReturnType The return type reflecting the size of the matrix to be filled. The actual result will be
   * a fixed typed matrix.
   * \tparam random_number_engine The random number engine.
   * \tparam Dists A set of distributions (e.g., std::normal_distribution<double>) or, alternatively,
   * means (a definite, non-stochastic value).
  **/
#ifdef __cpp_concepts
  template<typed_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename...Dists>
  requires (not has_dynamic_dimensions<ReturnType>) and (sizeof...(Dists) > 0) and
    (((requires { typename std::decay_t<Dists>::result_type;  typename std::decay_t<Dists>::param_type; } or
      std::is_arithmetic_v<std::decay_t<Dists>>) and ... )) and
    ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
    (sizeof...(Dists) == 1 or index_dimension_of_v<ReturnType, 0> * index_dimension_of_v<ReturnType, 1> == sizeof...(Dists) or
      index_dimension_of_v<ReturnType, 0> == sizeof...(Dists) or index_dimension_of_v<ReturnType, 1> == sizeof...(Dists))
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<typed_matrix<ReturnType> and (not has_dynamic_dimensions<ReturnType>) and (sizeof...(Dists) > 0) and
      ((not std::is_const_v<std::remove_reference_t<Dists>>) and ...) and
      (sizeof...(Dists) == 1 or
        index_dimension_of<ReturnType, 0>::value * index_dimension_of<ReturnType, 1>::value == sizeof...(Dists) or
        index_dimension_of<ReturnType, 0>::value == sizeof...(Dists) or
        index_dimension_of<ReturnType, 1>::value == sizeof...(Dists)), int> = 0>
#endif
  inline auto
  randomize(Dists&& ... dists)
  {
    using B = nested_object_of_t<ReturnType>;
    return MatrixTraits<std::decay_t<ReturnType>>::template make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }


  /**
   * \overload
   * \brief Fill a dynamic-shape typed_matrix with random values selected from a single random distribution.
   * \details The following example constructs two 2-by-2 matrices (m, n, and p) in which each element is a
   * random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     auto m = randomize<Matrix<Dimensions<2>, Dimensions<2>, Eigen::Matrix<float, 2, Eigen::Dynamic>>>(2, 2, std::normal_distribution<float> {1.0, 0.3}));
   *     auto n = randomize<Matrix<Dimensions<2>, Dimensions<2>, Eigen::Matrix<double, Eigen::Dynamic, 2>>>(2, 2, std::normal_distribution<double> {1.0, 0.3}));
   *     auto p = randomize<Matrix<Dimensions<2>, Dimensions<2>, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>>(2, 2, std::normal_distribution<double> {1.0, 0.3});
   *   \endcode
   * \tparam ReturnType The type of the matrix to be filled.
   * \tparam random_number_engine The random number engine (e.g., std::mt19937).
   * \param rows Number of rows, decided at runtime. Must match rows of ReturnType if they are fixed.
   * \param columns Number of columns, decided at runtime. Must match columns of ReturnType if they are fixed.
   * \tparam Dist A distribution (type distribution_type).
   **/
#ifdef __cpp_concepts
  template<typed_matrix ReturnType, std::uniform_random_bit_generator random_number_engine = std::mt19937,
    typename Dist>
  requires has_dynamic_dimensions<ReturnType> and
    requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; } and
    (not std::is_const_v<std::remove_reference_t<Dist>>)
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename Dist, std::enable_if_t<
      typed_matrix<ReturnType> and has_dynamic_dimensions<ReturnType> and
      (not std::is_const_v<std::remove_reference_t<Dist>>), int> = 0>
#endif
  inline auto
  randomize(const std::size_t rows, const std::size_t columns, Dist&& dist)
  {
    if constexpr (not dynamic_dimension<ReturnType, 0>) assert(rows == index_dimension_of_v<ReturnType, 0>);
    if constexpr (not dynamic_dimension<ReturnType, 1>) assert(columns == index_dimension_of_v<ReturnType, 1>);
    using B = nested_object_of_t<ReturnType>;
    return MatrixTraits<std::decay_t<ReturnType>>::template make(randomize<B, random_number_engine>(
      rows, columns, std::forward<Dist>(dist)));
  }


  /// Output the vector to a stream.
#ifdef __cpp_concepts
  template<typed_matrix V>
#else
  template<typename V, std::enable_if_t<typed_matrix<V>, int> = 0>
#endif
  inline std::ostream& operator<<(std::ostream& os, const V& v)
  {
    os << make_dense_object(v);
    return os;
  }


}

#endif //OPENKALMAN_TYPED_MATRIX_OVERLOADS_H