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

#ifndef OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
#define OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP

namespace OpenKalman::interface
{

#ifdef __cpp_concepts
  template<euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    element_gettable<nested_matrix_of_t<T>, I, Is...>
  struct GetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct GetElement<T, std::enable_if_t<euclidean_expr<T> and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_gettable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, Is...is)
    {
      if constexpr (row_coefficient_types_of_t<Arg>::axes_only)
      {
        return get_element(nested_matrix(std::forward<Arg>(arg)), i, is...);
      }
      else
      {
        using Coeffs = row_coefficient_types_of_t<Arg>;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
            return get_element(nested_matrix(std::forward<Arg>(arg)), row, is...);
        };
        if constexpr(to_euclidean_expr<Arg>)
        {
          return OpenKalman::internal::to_euclidean_coeff<Coeffs>(i, get_coeff);
        }
        else
        {
          return OpenKalman::internal::from_euclidean_coeff<Coeffs>(i, get_coeff);
        }
      }
    }
  };


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and to_euclidean_expr<nested_matrix_of_t<T>> and
    element_gettable<nested_matrix_of_t<T>, I, Is...>
  struct GetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct GetElement<T, std::enable_if_t<from_euclidean_expr<T> and to_euclidean_expr<nested_matrix_of_t<T>> and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_gettable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg>
    static constexpr auto get(Arg&& arg, I i, Is...is)
    {
      if constexpr (row_coefficient_types_of_t<Arg>::axes_only)
      {
        return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), i, is...);
      }
      else
      {
        using Coeffs = row_coefficient_types_of_t<Arg>;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
          return get_element(nested_matrix(nested_matrix(std::forward<Arg>(arg))), row, is...);
        };
        return OpenKalman::internal::wrap_get<Coeffs>(i, get_coeff);
      }
    }
  };


#ifdef __cpp_concepts
  template<euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
      row_coefficient_types_of_t<T>::axes_only and element_settable<nested_matrix_of_t<T>, I, Is...>
  struct SetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct SetElement<T, std::enable_if_t<euclidean_expr<T> and (not to_euclidean_expr<nested_matrix_of_t<T>>) and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    row_coefficient_types_of_t<T>::axes_only and
    (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    template<typename Arg, typename Scalar>
    static constexpr auto set(Arg&& arg, const Scalar s, I i, Is...is)
    {
      set_element(nested_matrix(arg), s, i, is...);
    }
  };


#ifdef __cpp_concepts
  template<from_euclidean_expr T, std::convertible_to<const std::size_t&> I, std::convertible_to<const std::size_t&>...Is>
    requires (sizeof...(Is) <= 1) and to_euclidean_expr<nested_matrix_of_t<T>> and
      element_settable<nested_matrix_of_t<T>, I, Is...>
  struct SetElement<T, I, Is...>
#else
  template<typename T, typename I, typename...Is>
  struct SetElement<T, std::enable_if_t<from_euclidean_expr<T> and to_euclidean_expr<nested_matrix_of_t<T>> and
    (std::is_convertible_v<I, const std::size_t&> and ... and std::is_convertible_v<Is, const std::size_t&>) and
    (sizeof...(Is) <= 1) and element_settable<nested_matrix_of_t<T>, I, Is...>>, I, Is...>
#endif
  {
    /**
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
    template<typename Arg, typename Scalar>
    static constexpr auto set(Arg&& arg, const Scalar s, I i, Is...is)
    {
      if constexpr (row_coefficient_types_of_t<Arg>::axes_only)
      {
        set_element(nested_matrix(nested_matrix(arg)), s, i, is...);
      }
      else
      {
        using Coeffs = row_coefficient_types_of_t<Arg>;
        const auto get_coeff = [&arg, is...] (const std::size_t row) {
          return get_element(nested_matrix(nested_matrix(arg)), row, is...);
        };
        const auto set_coeff = [&arg, is...] (const std::size_t row, const Scalar value) {
          set_element(nested_matrix(nested_matrix(arg)), value, row, is...);
        };
        OpenKalman::internal::wrap_set<Coeffs>(i, s, set_coeff, get_coeff);
      }
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
      if constexpr (uniform_coefficients<row_coefficient_types_of_t<Arg>> and
          row_coefficient_types_of_t<Arg>::dimension == row_coefficient_types_of_t<Arg>::euclidean_dimension)
      {
        using RC = typename has_uniform_coefficients<row_coefficient_types_of_t<Arg>>::common_coefficient;

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
  struct ArrayOperations<T>
#else
  template<typename T>
  struct ArrayOperations<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

    template<ElementOrder order, typename BinaryFunction, typename Accum, typename Arg>
    static constexpr auto fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
    {
      return OpenKalman::fold<order>(b, std::forward<Accum>(accum), make_dense_writable_matrix_from(std::forward<Arg>(arg)));
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
      if constexpr( row_coefficient_types_of_t<Arg>::axes_only)
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
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
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
  struct ModularTransformationTraits<T>
#else
  template<typename T>
  struct ModularTransformationTraits<T, std::enable_if_t<euclidean_expr<T>>>
#endif
  {

#ifdef __cpp_concepts
    template<typename...FC, from_euclidean_expr Arg, typename...DC>
#else
    template<typename...FC, typename Arg, typename...DC, std::enable_if_t<from_euclidean_expr<Arg>, int> = 0>
#endif
    constexpr decltype(auto)
    to_euclidean(Arg&& arg, DC&&...dc) noexcept
    {
      return nested_matrix(std::forward<Arg>(arg));
    }


#ifdef __cpp_concepts
    template<typename...FC, to_euclidean_expr Arg, typename...DC>
#else
    template<typename...FC, typename Arg, typename...DC, std::enable_if_t<to_euclidean_expr<Arg>, int> = 0>
#endif
    constexpr decltype(auto)
    from_euclidean(Arg&& arg, DC&&...dc) noexcept
    {
      return FromEuclideanExpr<FC..., DC..., Arg> {std::forward<Arg>(arg), std::forward<DC>(dc)...};
    }


#ifdef __cpp_concepts
    template<typename...FC, from_euclidean_expr Arg, typename...DC>
#else
    template<typename...FC, typename Arg, typename...DC, std::enable_if_t<from_euclidean_expr<Arg>, int> = 0>
#endif
    constexpr decltype(auto)
    wrap_angles(Arg&& arg, DC&&...dc) noexcept
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
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
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
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
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
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
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
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
      {
        return OpenKalman::determinant(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.determinant(); //< \todo Generalize this.
      }
    }


    template<typename Arg>
    static constexpr auto
    trace(Arg&& arg) noexcept
    {
      if constexpr(row_coefficient_types_of_t<Arg>::axes_only)
      {
        return OpenKalman::trace(nested_matrix(std::forward<Arg>(arg)));
      }
      else
      {
        return arg.trace(); //< \todo Generalize this.
      }
    }


    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto)
    rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_self_adjoint<t>(make_dense_writable_matrix_from(std::forward<A>(a)), std::forward<U>(u), alpha);
    }


    template<TriangleType t, typename A, typename U, typename Alpha>
    static decltype(auto) rank_update_triangular(A&& a, U&& u, const Alpha alpha)
    {
      return OpenKalman::rank_update_triangular<t>(make_dense_writable_matrix_from(std::forward<A>(a)), std::forward<U>(u), alpha);
    }


    template<bool must_be_unique, bool must_be_exact, typename A, typename B>
    static constexpr decltype(auto)
    solve(A&& a, B&& b) noexcept
    {
      return OpenKalman::solve<must_be_unique, must_be_exact>(
        to_native_matrix<T>(std::forward<A>(a)), std::forward<B>(b));
    }

  };


} // namespace OpenKalman::interface


namespace OpenKalman::Eigen3
{

#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_columns(Arg&& arg) noexcept
  {
    return make_dense_writable_matrix_from(reduce_columns(make_dense_writable_matrix_from(std::forward<Arg>(arg))));
  }


#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  constexpr auto
  reduce_rows(Arg&& arg) noexcept
  {
    return make_dense_writable_matrix_from(reduce_rows(make_dense_writable_matrix_from(std::forward<Arg>(arg))));
  }


/**
 * Perform an LQ decomposition of matrix A=[L,0]Q, L is a lower-triangular matrix, and Q is orthogonal.
 * Returns L as a triangular matrix.
 */
#ifdef __cpp_concepts
  template<euclidean_expr A>
#else
  template<typename A, std::enable_if_t<euclidean_expr<A>, int> = 0>
#endif
  inline auto
  LQ_decomposition(A&& a)
  {
    return LQ_decomposition(make_dense_writable_matrix_from(std::forward<A>(a)));
  }


  /**
   * Perform a QR decomposition of matrix A=Q[U,0], U is a upper-triangular matrix, and Q is orthogonal.
   * Returns U as an upper-triangular matrix.
   */
#ifdef __cpp_concepts
  template<euclidean_expr Arg>
#else
  template<typename Arg, std::enable_if_t<euclidean_expr<Arg>, int> = 0>
#endif
  inline auto
  QR_decomposition(Arg&& arg)
  {
    return QR_decomposition(make_dense_writable_matrix_from(std::forward<Arg>(arg)));
  }


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
      using C = Concatenate<row_coefficient_types_of_t<V>, row_coefficient_types_of_t<Vs>...>;
      return MatrixTraits<V>::template make<C>(
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
      return MatrixTraits<V>::template make<C>(
        concatenate_horizontal(nested_matrix(std::forward<V>(v)), nested_matrix(std::forward<Vs>(vs))...));
    }
    else
    {
      return std::forward<V>(v);
    }
  }


  namespace internal
  {
    template<typename G, typename Expr, typename CC>
    struct SplitEuclideanVertF
    {
      template<typename RC, typename, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };


    template<typename G, typename Expr, typename RC>
    struct SplitEuclideanHorizF
    {
      template<typename, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };


    template<typename G, typename Expr>
    struct SplitEuclideanDiagF
    {
      template<typename RC, typename CC, typename Arg>
      static auto call(Arg&& arg)
      {
        return G::template call<RC, CC>(MatrixTraits<Expr>::template make<RC>(std::forward<Arg>(arg)));
      }
    };
  } // internal


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    using CC = Axes<column_dimension_of_v<Arg>>;
    return split_vertical<internal::SplitEuclideanVertF<F, Arg, CC>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<typename F, bool, coefficients...Cs, euclidean_expr Arg> requires (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_vertical(Arg&& arg) noexcept
  {
    return split_vertical<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions vertically.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg> requires
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
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
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires
    (not coefficients<F>) and ((0 + ... + Cs::dimension) <= column_dimension_of_v<Arg>)
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    (not coefficients<F>) and ((0 + ... + Cs::dimension) <= column_dimension_of<Arg>::value), int> = 0>
#endif
  inline auto
  split_horizontal(Arg&& arg) noexcept
  {
    using RC = row_coefficient_types_of_t<Arg>;
    return split_horizontal<internal::SplitEuclideanHorizF<F, Arg, RC>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions horizontally.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<
    euclidean_expr<Arg> and (coefficients<Cs> and ...), int> = 0>
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
    return split_horizontal<Axes<cut>, Axes<cuts>...>(std::forward<Arg>(arg));
  }


  /**
   * \brief Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
   */
#ifdef __cpp_concepts
  template<typename F, coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<internal::SplitEuclideanDiagF<F, Arg>, from_euclidean_expr<Arg>, Cs...>(
      nested_matrix(std::forward<Arg>(arg)));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<typename F, bool, coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    (not coefficients<F>) and prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename F, bool, typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and
    square_matrix<Arg> and (not coefficients<F>) and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
#endif
  inline auto
  split_diagonal(Arg&& arg) noexcept
  {
    return split_diagonal<F, Cs...>(std::forward<Arg>(arg));
  }


  /// Split into one or more Euclidean expressions diagonally. The valuated expression must be square.
#ifdef __cpp_concepts
  template<coefficients...Cs, euclidean_expr Arg> requires square_matrix<Arg> and
    prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>
#else
  template<typename...Cs, typename Arg, std::enable_if_t<euclidean_expr<Arg> and square_matrix<Arg> and
    (coefficients<Cs> and ...) and prefix_of<Concatenate<Cs...>, row_coefficient_types_of_t<Arg>>, int> = 0>
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
    using Coefficients = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto& col)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(std::move(col));
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
    using Coefficients = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto& col, std::size_t i)
    {
      auto mc = MatrixTraits<Arg>::template make<Coefficients>(std::move(col));
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
    using Coefficients = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto&& col) -> auto {
      return make_self_contained(f(MatrixTraits<Arg>::template make<Coefficients>(std::forward<decltype(col)>(col))));
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
    using Coefficients = row_coefficient_types_of_t<Arg>;
    const auto f_nested = [&f](auto&& col, std::size_t i) -> auto {
      return make_self_contained(
        f(MatrixTraits<Arg>::template make<Coefficients>(std::forward<decltype(col)>(col)), i));
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
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
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
    return MatrixTraits<ResultType>::make(apply_columnwise<count>(f_nested));
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
    return MatrixTraits<ReturnType>::make(randomize<B, random_number_engine>(std::forward<Dists>(dists)...));
  }

} // namespace OpenKalman::Eigen3

#endif //OPENKALMAN_EIGEN3_EUCLIDEAN_OVERLOADS_HPP
