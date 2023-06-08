/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2020-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded functions relating to Eigen3::euclidean_expr types
 */

#ifndef OPENKALMAN_EUCLIDEAN_OVERLOADS_HPP
#define OPENKALMAN_EUCLIDEAN_OVERLOADS_HPP

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct GetElement<T>
#else
  template<typename T>
  struct GetElement<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {
#ifdef __cpp_lib_concepts
    template<typename Arg, typename I, typename...Is> requires element_gettable<nested_matrix_of_t<Arg>, I, Is...>
#else
    template<typename Arg, typename I, typename...Is, std::enable_if_t<element_gettable<typename nested_matrix_of<Arg>::type, I, Is...>, int> = 0>
#endif
    static constexpr auto get(Arg&& arg, I i, Is...is)
    {
      if constexpr (has_untyped_index<Arg, 0>)
      {
        if constexpr (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>)
          return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i, is...);
        else
          return get_element(nested_matrix(std::forward<Arg>(arg)), i, is...);
      }
      else
      {
        auto g {[&arg, is...](std::size_t ix) { return get_element(nested_matrix(std::forward<Arg>(arg)), ix, is...); }};
        if constexpr (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>)
          return wrap_get_element(get_dimensions_of<0>(arg), g, i, 0);
        else if constexpr(to_euclidean_expr<Arg>)
          return to_euclidean_element(get_dimensions_of<0>(arg), g, i, 0);
        else
          return from_euclidean_element(get_dimensions_of<0>(arg), g, i, 0);
      }
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct SetElement<T>
#else
  template<typename T>
  struct SetElement<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {
    /**
     * \internal
     * \brief Set element (i, j) of arg in FromEuclideanExpr(ToEuclideanExpr(arg)) to s.
     * \details This function sets the nested matrix, not the wrapped resulting matrix.
     * For example, if the coefficient is Polar<Distance, angle::Radians> and the initial value of a
     * single-column vector is {-1., pi/2}, then set_element(arg, pi/4, 1, 0) will replace p/2 with pi/4 to
     * yield {-1., pi/4} in the nested matrix. The resulting wrapped expression will yield {1., -3*pi/4}.
     * \tparam Arg The matrix to set.
     * \tparam Scalar The value to set the coefficient to.
     * \param i The row of the coefficient.
     * \param j The column of the coefficient.
     */
#ifdef __cpp_lib_concepts
    template<typename Arg, typename I, typename...Is> requires element_gettable<nested_matrix_of_t<Arg>, I, Is...> and
      (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>))
#else
    template<typename Arg, typename I, typename...Is, std::enable_if_t<
      element_gettable<typename nested_matrix_of<Arg>::type, I, Is...> and
      (has_untyped_index<Arg, 0> or (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>)), int> = 0>
#endif
    static constexpr Arg&& set(Arg&& arg, const scalar_type_of_t<Arg>& s, I i, Is...is)
    {
      if constexpr (has_untyped_index<Arg, 0>)
      {
        set_element(nested_matrix(nested_matrix(arg)), s, i, is...);
      }
      else if constexpr (from_euclidean_expr<Arg> and to_euclidean_expr<nested_matrix_of_t<Arg>>)
      {
        auto s {[&arg, is...](const scalar_type_of_t<Arg>& x, std::size_t i) {
          return set_element(nested_matrix(nested_matrix(arg)), x, i, is...);
        }};
        auto g {[&arg, is...](std::size_t ix) {
          return get_element(nested_matrix(nested_matrix(arg)), ix, is...);
        }};
        wrap_set_element(get_dimensions_of<0>(arg), s, g, s, i, 0);
      }
      else
      {
        set_element(nested_matrix(arg), s, i, is...);
      }

      return std::forward<Arg>(arg);
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct Subsets<T>
#else
  template<typename T>
  struct Subsets<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {
    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    column(Arg&& arg, runtime_index_t...i)
    {
      using RC = row_coefficient_types_of_t<Arg>;
      if constexpr (from_euclidean_expr<Arg>)
        return from_euclidean<RC>(column<compile_time_index...>(nested_matrix(std::forward<Arg>(arg)), i...));
      else
        return to_euclidean<RC>(column<compile_time_index...>(nested_matrix(std::forward<Arg>(arg)), i...));
    }


    template<std::size_t...compile_time_index, typename Arg, typename...runtime_index_t>
    static constexpr decltype(auto)
    row(Arg&& arg, runtime_index_t...i)
    {
      if constexpr (has_uniform_dimension_type<row_coefficient_types_of_t<Arg>> and
        dimension_size_of_v<row_coefficient_types_of_t<Arg>> == euclidean_dimension_size_of_v<row_coefficient_types_of_t<Arg>>)
      {
        using RC = uniform_dimension_type_of_t<row_coefficient_types_of_t<Arg>>;

        if constexpr (from_euclidean_expr<Arg>)
          return from_euclidean<RC>(row<compile_time_index...>(nested_matrix(std::forward<Arg>(arg)), i...));
        else
          return to_euclidean<RC>(row<compile_time_index...>(nested_matrix(std::forward<Arg>(arg)), i...));
      }
      else
      {
        return row<compile_time_index...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)), i...);
      }
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct Conversions<T>
#else
  template<typename T>
  struct Conversions<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<typename Arg>
    static auto
    to_diagonal(Arg&& arg) noexcept
    {
      if constexpr( has_untyped_index<Arg, 0>)
      {
        return to_diagonal(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        using P = pattern_matrix_of_t<T>;
        return Conversions<P>::to_diagonal(to_native_matrix<P>(std::forward<Arg>(arg)));
      }
    }


    template<typename Arg>
    static auto
    diagonal_of(Arg&& arg) noexcept
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return diagonal_of(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        using P = pattern_matrix_of_t<T>;
        return Conversions<P>::diagonal_of(to_native_matrix<P>(std::forward<Arg>(arg)));
      }
    }

  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct ArrayOperations<T>
#else
  template<typename T>
  struct ArrayOperations<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<typename...Ds, typename Operation, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation(const std::tuple<Ds...>& tup, Operation&& op, Args&&...args)
    {
      using P = pattern_matrix_of_t<T>;
      return ArrayOperations<P>::template n_ary_operation(tup, std::forward<Operation>(op), std::forward<Args>(args)...);
    }


    template<std::size_t...indices, typename BinaryFunction, typename Arg>
    static constexpr decltype(auto)
    reduce(BinaryFunction&& b, Arg&& arg)
    {
      using P = pattern_matrix_of_t<T>;
      return ArrayOperations<P>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }

  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct ModularTransformationTraits<T>
#else
  template<typename T>
  struct ModularTransformationTraits<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

#ifdef __cpp_concepts
    template<from_euclidean_expr Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<from_euclidean_expr<Arg> and index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, const C&) noexcept
    {
      return nested_matrix(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<to_euclidean_expr Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<to_euclidean_expr<Arg> and index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, const C& c) noexcept
    {
      return FromEuclideanExpr<C, Arg> {std::forward<Arg>(arg), c};
    }


#ifdef __cpp_concepts
    template<from_euclidean_expr Arg, index_descriptor C>
#else
    template<typename Arg, typename C, std::enable_if_t<from_euclidean_expr<Arg> and index_descriptor<C>, int> = 0>
#endif
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, const C&) noexcept
    {
      return std::forward<Arg>(arg);
    }

  };


#ifdef __cpp_concepts
  template<euclidean_expr T>
  struct LinearAlgebra<T>
#else
  template<typename T>
  struct LinearAlgebra<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<typename Arg>
    static constexpr decltype(auto)
    conjugate(Arg&& arg) noexcept
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::conjugate(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).conjugate(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    transpose(Arg&& arg) noexcept
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::transpose(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).transpose(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr decltype(auto)
    adjoint(Arg&& arg) noexcept
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::adjoint(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return std::forward<Arg>(arg).adjoint(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr auto
    determinant(Arg&& arg) noexcept
    {
      if constexpr(has_untyped_index<Arg, 0>)
      {
        return OpenKalman::determinant(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.determinant(); //< \todo Generalize this.
      }
    }


    template<HermitianAdapterType significant_triangle, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_self_adjoint<significant_triangle>(make_hermitian_matrix(make_dense_writable_matrix_from(std::forward<A>(a))), std::forward<U>(u), alpha);
    }


    template<TriangleType triangle, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_triangular(make_triangular_matrix<triangle>(make_dense_writable_matrix_from(std::forward<A>(a))), std::forward<U>(u), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b) noexcept
    {
      return OpenKalman::solve<must_be_unique, must_be_exact>(
        to_native_matrix<T>(std::forward<A>(a)), std::forward<B>(b));
    }


    template<typename A>
    static inline auto
    LQ_decomposition(A&& a)
    {
      return LQ_decomposition(make_dense_writable_matrix_from(std::forward<A>(a)));
    }


    template<typename A>
    static inline auto
    QR_decomposition(A&& a)
    {
      return QR_decomposition(make_dense_writable_matrix_from(std::forward<A>(a)));
    }

  };


} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

  /// Concatenate one or more EuclideanExpr objects vertically.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_vertical(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      constexpr auto cols = column_dimension_of_v<V>;
      static_assert(((cols == column_dimension_of_v<Vs>) and ...));
      using C = concatenate_fixed_index_descriptor_t<row_coefficient_types_of_t<V>, row_coefficient_types_of_t<Vs>...>;
      return MatrixTraits<std::decay_t<V>>::template make<C>(
        concatenate_vertical(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  /// Concatenate one or more EuclideanExpr objects horizontally.
#ifdef __cpp_concepts
  template<typename V, typename ... Vs> requires (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>)
#else
  template<typename V, typename ... Vs, std::enable_if_t<
    (to_euclidean_expr<V> and ... and to_euclidean_expr<Vs>) or
    (from_euclidean_expr<V> and ... and from_euclidean_expr<Vs>), int> = 0>
#endif
  constexpr decltype(auto)
  concatenate_horizontal(V&& v, Vs&& ... vs) noexcept
  {
    if constexpr (sizeof...(Vs) > 0)
    {
      using C = row_coefficient_types_of_t<V>;
      static_assert(std::conjunction_v<std::is_same<C, row_coefficient_types_of_t<Vs>>...>);
      return MatrixTraits<std::decay_t<V>>::template make<C>(
        concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  template<typename G, typename Expr, typename CC>
  struct SplitEuclideanVertF
  {
    template<typename RC, typename, typename Arg>
    static auto call(Arg&& arg)
    {
      return G::template call<RC, CC>(MatrixTraits<std::decay_t<Expr>>::template make<RC>(std::forward<Arg>(arg)));
    }
  };


  template<typename G, typename Expr, typename RC>
  struct SplitEuclideanHorizF
  {
    template<typename, typename CC, typename Arg>
    static auto call(Arg&& arg)
    {
      return G::template call<RC, CC>(MatrixTraits<std::decay_t<Expr>>::template make<RC>(std::forward<Arg>(arg)));
    }
  };


  template<typename G, typename Expr>
  struct SplitEuclideanDiagF
  {
    template<typename RC, typename CC, typename Arg>
    static auto call(Arg&& arg)
    {
      return G::template call<RC, CC>(MatrixTraits<std::decay_t<Expr>>::template make<RC>(std::forward<Arg>(arg)));
    }
  };


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor...Cs, euclidean_expr Arg> requires (not fixed_index_descriptor<F>) and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not fixed_index_descriptor<F>) and prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    using CC = Dimensions<column_dimension_of_v<Arg>>;
    return split_vertical<SplitEuclideanVertF<F, Arg, CC>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, bool, fixed_index_descriptor...Cs, euclidean_expr Arg> requires (not fixed_index_descriptor<F>) and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not fixed_index_descriptor<F>) and prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<fixed_index_descriptor...Cs, euclidean_expr Arg> requires
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions vertically.
   * \details The expression is evaluated to a self_contained matrix first.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Number of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires
    ((cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and ((cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    if constexpr(cut == row_dimension_of_v<Arg> and sizeof...(cuts) == 0)
    {
      return std::tuple<Arg> {std::forward<Arg>(arg)};
    }
    else
    {
      return split_vertical<cut, cuts...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
    }
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor...Cs, euclidean_expr Arg> requires
    (not fixed_index_descriptor<F>) and ((0 + ... + dimension_size_of_v<Cs>) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not fixed_index_descriptor<F>) and ((0 + ... + dimension_size_of_v<Cs>) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    using RC = row_coefficient_types_of_t<Arg>;
    return split_horizontal<SplitEuclideanHorizF<F, Arg, RC>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<fixed_index_descriptor...Cs, euclidean_expr Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and (fixed_index_descriptor<Cs> and ...), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<OpenKalman::internal::default_split_function, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions horizontally.
   * \tparam cut Number of columns in the first cut.
   * \tparam cuts Number of columns in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires
    ((cut + ... + cuts) <= column_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    ((cut + ... + cuts) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    return split_horizontal<Dimensions<cut>, Dimensions<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
   */
#ifdef __cpp_concepts
  template<typename F, fixed_index_descriptor...Cs, euclidean_expr Arg> requires square_matrix<Arg> and (not fixed_index_descriptor<F>) and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (not fixed_index_descriptor<F>) and prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<SplitEuclideanDiagF<F, Arg>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<typename F, bool, fixed_index_descriptor...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    (not fixed_index_descriptor<F>) and prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    square_matrix<Arg> and (not fixed_index_descriptor<F>) and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<fixed_index_descriptor...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (fixed_index_descriptor<Cs> and ...) and prefix_of<concatenate_fixed_index_descriptor_t<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<OpenKalman::internal::default_split_function, false, Cs...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally.
   * |details The expression (which must be square) is evaluated to a self_contained matrix first.
   * \tparam cut Number of rows in the first cut.
   * \tparam cuts Number of rows in the second and subsequent cuts.
   */
#ifdef __cpp_concepts
  template<std::size_t cut, std::size_t ... cuts, euclidean_expr Arg> requires square_matrix<Arg> and
    ((cut + ... + cuts) <= row_dimension_of_v<Arg>)
#else
  template<std::size_t cut, std::size_t ... cuts, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and square_matrix<Arg> and ((cut + ... + cuts) <= row_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    if constexpr(cut == row_dimension_of_v<Arg> and sizeof...(cuts) == 0)
    {
      return std::tuple<Arg> {std::forward<Arg>(arg)};
    }
    else
    {
      return split_diagonal<cut, cuts...>(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
    }
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg An lvalue reference to the matrix.
   * \tparam Function The function, which takes a column and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col) { f(col); } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    std::is_invocable_v<Function, decltype(column(std::declval<Arg&>(), 0))&> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using TypedIndex = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<std::decay_t<Arg>>::template make<TypedIndex>(std::move(col));
      f(mc);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(f_nested, c);
    return arg;
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg An lvalue reference to the matrix.
   * \tparam Function The function, which takes a column and an index and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, std::decay_t<decltype(column(std::declval<Arg>(), 0))>& col, std::size_t i) {
      f(col, i);
    } and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    std::is_invocable_v<Function, decltype(column(std::declval<Arg&>(), 0))&, std::size_t> and
    (not std::is_const_v<std::remove_reference_t<nested_matrix_of_t<Arg>>>) and
    modifiable<nested_matrix_of_t<Arg>, nested_matrix_of_t<Arg>>, int> = 0>
#endif
  inline Arg&
  apply_columnwise(const Function& f, Arg& arg)
  {
    using TypedIndex = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<std::decay_t<Arg>>::template make<TypedIndex>(std::move(col));
      f(mc, i);
      col = std::move(nested_matrix(mc));
    };
    auto& c = nested_matrix(arg);
    apply_columnwise(f_nested, c);
    return arg;
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a column and returns a column.
   * \todo add situation where Arg is a native matrix but the function result is a euclidean_expr
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Arg& arg, const Function& f) { {f(column(arg, 0))} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    column_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    using TypedIndex = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(f(MatrixTraits<std::decay_t<Arg>>::template make<TypedIndex>(std::forward<decltype(col)>(col))));
    };
    return apply_columnwise(f_nested, nested_matrix(arg));
  }


  /**
   * \brief Apply a function to each column of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a column and an index and returns a column.
   */
#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Arg& arg, const Function& f, std::size_t i) { {f(column(arg, 0), i)} -> column_vector; }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    column_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(column(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f, const Arg& arg)
  {
    using TypedIndex = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      return make_self_contained(
        f(MatrixTraits<std::decay_t<Arg>>::template make<TypedIndex>(std::forward<decltype(col)>(col)), i));
    };
    return apply_columnwise(f_nested, nested_matrix(arg));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> euclidean_expr; {f()} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function>> and column_vector<std::invoke_result_t<Function>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function>;
    const auto f_nested = [&f] () -> auto { return make_self_contained(nested_matrix(f())); };
    return MatrixTraits<std::decay_t<ResultType>>::make(apply_columnwise<count>(f_nested));
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> euclidean_expr; {f(i)} -> column_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function, std::size_t>> and
    column_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_columnwise(const Function& f)
  {
    using ResultType = std::invoke_result_t<Function, std::size_t>;
    const auto f_nested = [&f](std::size_t i) -> auto { return make_self_contained(nested_matrix(f(i))); };
    return MatrixTraits<std::decay_t<ResultType>>::make(apply_columnwise<count>(f_nested));
  }


  /**
   * \brief Apply a function to each row of a matrix.
   * \tparam Arg A constant lvalue reference or rvalue reference to the matrix.
   * \tparam Function The function, which takes a row (and optionally an index) and returns a row.
   */
  #ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(const Function& f, const Arg& arg) { {f(row(arg, 0))} -> row_vector; } or
    requires(const Function& f, const Arg& arg, std::size_t i) { {f(row(arg, 0), i)} -> row_vector; }
  #else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    row_vector<std::invoke_result_t<Function, std::decay_t<decltype(row(std::declval<const Arg&>(), 0))>&&>>, int> = 0>
  inline auto
  apply_rowwise(const Function& f, Arg&& arg)
  {
    return apply_rowwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }

  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    row_vector<std::invoke_result_t<Function,
      std::decay_t<decltype(row(std::declval<const Arg&>(), 0))>&&, std::size_t>>, int> = 0>
  #endif
  inline auto
  apply_rowwise(const Function& f, Arg&& arg)
  {
    return apply_rowwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }


  #ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f) { {f()} -> euclidean_expr; {f()} -> row_vector; }
  #else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function>> and row_vector<std::invoke_result_t<Function>>, int> = 0>
  #endif
  inline auto
  apply_rowwise(const Function& f)
  {
    const auto f_nested = [&f] () -> auto { return make_dense_writable_matrix_from(f()); };
    return apply_rowwise<count>(f_nested);
  }


#ifdef __cpp_concepts
  template<std::size_t count, typename Function> requires
    requires(const Function& f, std::size_t i) { {f(i)} -> euclidean_expr; {f(i)} -> row_vector; }
#else
  template<std::size_t count, typename Function, std::enable_if_t<
    euclidean_expr<std::invoke_result_t<Function, std::size_t>> and
      row_vector<std::invoke_result_t<Function, std::size_t>>, int> = 0>
#endif
  inline auto
  apply_rowwise(const Function& f)
  {
    const auto f_nested = [&f](std::size_t i) -> auto { return make_dense_writable_matrix_from(f(i)); };
    return apply_rowwise<count>(f_nested);
  }


#ifdef __cpp_concepts
  template<typename Function, euclidean_expr Arg> requires
    requires(Function& f, scalar_type_of_t<Arg>& s) {
      {f(s)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    } or
    requires(Function& f, scalar_type_of_t<Arg>& s, std::size_t& i, std::size_t& j) {
      {f(s, i, j)} -> std::convertible_to<const scalar_type_of_t<Arg>>;
    }
#else
  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and std::is_convertible_v<
    std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }


  template<typename Function, typename Arg, std::enable_if_t<euclidean_expr<Arg> and std::is_convertible_v<
    std::invoke_result_t<Function&, typename scalar_type_of<Arg>::type&, std::size_t&, std::size_t&>,
    const typename scalar_type_of<Arg>::type>, int> = 0>
#endif
  inline auto
  apply_coefficientwise(const Function& f, Arg&& arg)
  {
    return apply_coefficientwise(f, make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }


  /**
   * \brief Fill a matrix of to-Euclidean- or from_euclidean-transformed values selected from a random distribution.
   * \details The Gaussian distribution has zero mean and standard deviation sigma (1, if not specified).
   **/
#ifdef __cpp_concepts
  template<euclidean_expr ReturnType,
    std::uniform_random_bit_generator random_number_engine = std::mt19937, typename...Dists>
#else
  template<typename ReturnType, typename random_number_engine = std::mt19937, typename...Dists,
    std::enable_if_t<euclidean_expr<ReturnType>, int> = 0>
#endif
  inline auto
  randomize(Dists...dists)
  {
    using B = nested_matrix_of_t<ReturnType>;
    return MatrixTraits<std::decay_t<ReturnType>>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EUCLIDEAN_OVERLOADS_HPP
