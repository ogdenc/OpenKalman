/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef OPENKALMAN_FUNCTIONS_HPP
#define OPENKALMAN_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // -------------------- //
  //  make_native_matrix  //
  // -------------------- //

  /**
   * \brief Convert to an equivalent, dense, writable matrix.
   */
#ifdef __cpp_concepts
  template<typename Arg> requires requires { typename EquivalentDenseWritableMatrix<Arg>; }
#else
  template<typename Arg, typename = std::void_t<interface::EquivalentDenseWritableMatrix<Arg>>>
#endif
  constexpr decltype(auto)
  make_native_matrix(Arg&& arg) noexcept
  {
    using Trait = EquivalentDenseWritableMatrix<std::decay_t<Arg>>;
    using T = std::remove_reference_t<decltype(Trait::convert(std::declval<Arg&&>()))>;
    static_assert(not std::is_const_v<T>, "EquivalentDenseWritableMatrix::convert() logic error: returns const result");
    if constexpr (std::is_same_v<std::remove_reference_t<Arg>, T>)
      return std::forward<Arg>(arg);
    else
      return Trait::convert(std::forward<Arg>(arg));
  }


#ifdef __cpp_concepts
  template<typename M, std::convertible_to<const scalar_type_of_t<M>> ... Args>
  requires ((row_extent_of_v<M> == 0 and column_extent_of_v<M> == 0) or
    (row_extent_of_v<M> != 0 and
      column_extent_of_v<M> != 0 and sizeof...(Args) == row_extent_of_v<M> * column_extent_of_v<M>) or
    (column_extent_of_v<M> == 0 and sizeof...(Args) % row_extent_of_v<M> == 0) or
    (row_extent_of_v<M> == 0 and sizeof...(Args) % column_extent_of_v<M> == 0)) and
    requires { typename MatrixTraits<std::decay_t<decltype(
      interface::EquivalentDenseWritableMatrix<std::decay_t<M>>::convert(std::declval<M>()))>>; }
#else
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wdiv-by-zero"
  template<typename M, typename ... Args, std::enable_if_t<
    (std::is_convertible_v<Args, const typename scalar_type_of<M>::type> and ...) and
    ((row_extent_of<M>::value == 0 and column_extent_of<M>::value == 0) or
    (row_extent_of<M>::value != 0 and
      column_extent_of<M>::value != 0 and sizeof...(Args) == row_extent_of<M>::value * column_extent_of<M>::value) or
    (column_extent_of<M>::value == 0 and sizeof...(Args) % row_extent_of<M>::value == 0) or
    (row_extent_of<M>::value == 0 and sizeof...(Args) % column_extent_of<M>::value == 0)), int> = 0,
    typename = std::void_t<MatrixTraits<std::decay_t<decltype(
          interface::EquivalentDenseWritableMatrix<std::decay_t<M>>::convert(std::declval<M>()))>>>>
#endif
  inline auto
  make_native_matrix(const Args ... args)
  {
    using Nat = std::decay_t<decltype(
          interface::EquivalentDenseWritableMatrix<std::decay_t<M>>::convert(std::declval<M>()))>;
    return MatrixTraits<Nat>::make(static_cast<const scalar_type_of_t<M>>(args)...);
  }
#ifndef __cpp_concepts
#   pragma GCC diagnostic pop
#endif


  // --------------- //
  //  nested_matrix  //
  // --------------- //

  /**
   * \brief Retrieve a nested matrix of Arg, if it exists.
   * \tparam i Index of the nested matrix (0 for the 1st, 1 for the 2nd, etc.).
   * \tparam Arg A wrapper that has at least one nested matrix.
   * \internal \sa interface::Dependencies::get_nested_matrix
   */
#ifdef __cpp_concepts
  template<std::size_t i = 0, typename Arg> requires
    (i < std::tuple_size_v<typename Dependencies<std::decay_t<Arg>>::type>) and
    requires(Arg&& arg) { Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg)); }
#else
  template<std::size_t i = 0, typename Arg,
    std::enable_if_t<(i < std::tuple_size_v<typename Dependencies<std::decay_t<Arg>>::type)>, int> == 0,
    typename = std::void_t<decltype(Dependencies<std::decay_t<T>>::template get_nested_matrix<i>(std::declval<Arg&&>()))>>
#endif
  constexpr decltype(auto)
  nested_matrix(Arg&& arg) noexcept
  {
      return Dependencies<std::decay_t<Arg>>::template get_nested_matrix<i>(std::forward<Arg>(arg));
  }


  // --------------------- //
  //  make_self_contained  //
  // --------------------- //

  namespace detail
  {
    template<typename Tup, std::size_t...I>
    constexpr bool all_lvalue_ref_dependencies_impl(std::index_sequence<I...>)
    {
      return ((sizeof...(I) > 0) and ... and std::is_lvalue_reference_v<std::tuple_element_t<I, Tup>>);
    }

#ifdef __cpp_concepts
    template<typename T>
    concept all_lvalue_ref_dependencies = (not Dependencies<std::decay_t<T>>::has_runtime_parameters) and
      all_lvalue_ref_dependencies_impl<typename Dependencies<std::decay_t<T>>::type>(
        std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<T>>::type>> {});
#else
    template<typename T, typename = void>
    struct has_no_runtime_parameters_impl : std::false_type {};

    template<typename T>
    struct has_no_runtime_parameters_impl<T, std::enable_if_t<not Dependencies<T>::has_runtime_parameters>>
      : std::true_type {};


    template<typename T, typename = void>
    struct all_lvalue_ref_dependencies_detail : std::false_type {};

    template<typename T>
    struct all_lvalue_ref_dependencies_detail<T, std::void_t<typename Dependencies<T>::type>>
      : std::bool_constant<has_no_runtime_parameters_impl<T> and
        all_lvalue_ref_dependencies_impl<typename Dependencies<T>::type>(
          std::make_index_sequence<std::tuple_size_v<typename Dependencies<T>::type>> {})> {};

    template<typename T>
    constexpr bool all_lvalue_ref_dependencies = all_lvalue_ref_dependencies_detail<std::decay_t<T>>::value;


    template<typename T, typename = void>
    struct convert_to_self_contained_is_defined : std::false_type {};

    template<typename T>
    struct convert_to_self_contained_is_defined<T,
      std::void_t<decltype(Dependencies<std::decay_t<T>>::convert_to_self_contained(std::declval<T&&>()))>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Convert to a self-contained version of Arg that can be returned in a function.
   * \details If any types Ts are included, Arg will not be converted to a self-contained version if every Ts is either
   * an lvalue reference or has a nested matrix that is an lvalue reference. This is to allow a function, taking Ts...
   * as lvalue-reference inputs or as rvalue-reference inputs that nest lvalue-references to other matrices, to avoid
   * unnecessary conversion because the referenced objects are accessible outside the scope of the function and do not
   * result in dangling references.
   * The following example adds two matrices arg1 and arg2 together and returns a self-contained matrix, unless
   * <em>both</em> Arg1 and Arg2 are lvalue references or their nested matrices are lvalue references, in which case
   * the result of the addition is returned without eager evaluation:
   * \code
   *   template<typename Arg1, typename Arg2>
   *   static decltype(auto) add(Arg1&& arg1, Arg2&& arg2)
   *   {
   *     return make_self_contained<Arg1, Arg2>(arg1 + arg2);
   *   }
   * \endcode
   * \tparam Ts Generally, these will be forwarding-reference arguments to the directly enclosing function. If all of
   * Ts... are lvalue references, Arg is returned without modification (i.e., without any potential eager evaluation).
   * \tparam Arg The potentially non-self-contained argument to be converted
   * \return A self-contained version of Arg (if it is not already self-contained)
   * \internal \sa interface::Dependencies
   */
  template<typename ... Ts, typename Arg>
  constexpr decltype(auto)
  make_self_contained(Arg&& arg) noexcept
  {
    if constexpr (self_contained<Arg> or
      ((sizeof...(Ts) > 0) and ... and (std::is_lvalue_reference_v<Ts> or detail::all_lvalue_ref_dependencies<Ts>)))
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (self_contained<std::remove_reference_t<Arg>>)
    {
      // If it's not self-contained because it is an lvalue reference, simply make a copy and return by value.
      return std::remove_reference_t<Arg> {std::forward<Arg>(arg)};
    }
#ifdef __cpp_concepts
    else if constexpr (requires {Dependencies<std::decay_t<Arg>>::convert_to_self_contained(std::forward<Arg>(arg)); })
#else
    else if constexpr (detail::convert_to_self_contained_is_defined<Arg>::value)
#endif
    {
      return Dependencies<std::decay_t<Arg>>::convert_to_self_contained(std::forward<Arg>(arg));
    }
    else
    {
      return make_native_matrix(std::forward<Arg>(arg));
    }
  }


  // ============================== //
  //  Element and extent functions  //
  // ============================== //

  // \todo generalize RowExtentOf and ColumnExtentOf for arbitrarily dimensioned arrays
  template<typename Arg>
  constexpr std::size_t row_count(Arg&& arg)
  {
    return interface::RowExtentOf<std::decay_t<Arg>>::rows_at_runtime(std::forward<Arg>(arg));
  }


  template<typename Arg>
  constexpr std::size_t column_count(Arg&& arg)
  {
    return interface::ColumnExtentOf<std::decay_t<Arg>>::columns_at_runtime(std::forward<Arg>(arg));
  }


  template<std::size_t N, typename Arg>
  constexpr std::size_t runtime_extent(Arg&& arg)
  {
    if constexpr (N == 0)
    {
      return row_count(std::forward<Arg>(arg));
    }
    else
    {
      static_assert(N == 1);
      return column_count(std::forward<Arg>(arg));
    }
  }


  namespace detail
  {
    template<bool set, typename Arg, std::size_t...seq, typename...I>
    inline void check_index_bounds(const Arg& arg, std::index_sequence<seq...>, I...i)
    {
      if constexpr (sizeof...(I) == 1)
      {
        auto c = column_count(arg);
        if (c == 1)
        {
          auto r = row_count(arg);
          if ((static_cast<std::size_t>(i),...) >= r)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Row index (which is " +
            std::to_string((i,...)) + ") is not in range 0 <= i < " + std::to_string(r) + ".")};
        }
        else
        {
          if ((static_cast<std::size_t>(i),...) >= c)
            throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + " Column index (which is " +
            std::to_string((i,...)) + ") is not in range 0 <= i < " + std::to_string(c) + ".")};
        }
      }
      else
      {
        (((static_cast<std::size_t>(i) >= runtime_extent<seq>(arg)) ?
          throw std::out_of_range {((std::string {set ? "s" : "g"} + "et_element:") + ... +
            (" Index " + std::to_string(seq) + " (which is " + std::to_string(i) + ") is not in range 0 <= i < " +
            std::to_string(runtime_extent<seq>(arg)) + "."))} :
          false) , ...);
      }
    }
  }


  /// Get element of matrix arg using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<std::size_t>...I> requires
    element_gettable<Arg, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
#else
  template<typename Arg, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
    element_gettable<Arg, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...> and
    (sizeof...(I) != 1 or column_vector<Arg> or row_vector<Arg>), int> = 0>
#endif
  constexpr auto get_element(Arg&& arg, const I...i)
  {
    detail::check_index_bounds<false>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::GetElement<std::decay_t<Arg>, I...>::get(std::forward<Arg>(arg), i...);
  }


  /// Set element to s using I... indices.
#ifdef __cpp_concepts
  template<typename Arg, std::convertible_to<const scalar_type_of_t<Arg>&> Scalar, std::convertible_to<std::size_t>...I>
    requires element_settable<Arg&, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>
#else
  template<typename Arg, typename Scalar, std::enable_if_t<(std::is_convertible_v<I, std::size_t> and ...) and
    std::is_convertible_v<Scalar, const scalar_type_of_t<Arg>&> and
    element_settable<Arg&, std::conditional_t<std::same_as<I, std::size_t>, I, std::size_t>...>, int> = 0>
#endif
  inline void set_element(Arg& arg, Scalar s, const I...i)
  {
    detail::check_index_bounds<true>(arg, std::make_index_sequence<sizeof...(I)>{}, i...);
    return interface::SetElement<std::decay_t<Arg>, I...>::set(arg, s, i...);
  }


  // ========================= //
  //  Element-wise operations  //
  // ========================= //

  /**
   * \brief Fold an operation across the elements of a matrix
   * \detail BinaryFunction must be invocable with two values, the first an accumulator and the second of type
   * <code>scalar_type_of_t<Arg></code>. It returns the accumulated value. After each iteration, the result of the
   * operation is used as the accumulator for the next iteration.
   * \tparam BinaryFunction A binary function (e.g. std::plus, std::multiplies)
   * \tparam Accum An accumulator
   * \tparam order The element order over which to perform the operation
   */
#ifdef __cpp_concepts
  template<ElementOrder order = ElementOrder::column_major, typename BinaryFunction, typename Accum, typename Arg>
    requires std::is_invocable_r_v<const std::remove_reference_t<Accum>&, const BinaryFunction&,
      Accum&&, scalar_type_of_t<Arg>> and std::move_constructible<std::decay_t<Accum>> and
    std::copy_constructible<std::decay_t<Accum>>
  constexpr decltype(auto) fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
#else
  template<ElementOrder order = ElementOrder::column_major, typename BinaryFunction, typename Accum, typename Arg,
    std::enable_if_t<std::is_invocable<const std::remove_reference_t<Accum>&, const BinaryFunction&,
      Accum&&, scalar_type_of_t<Arg>>::value and std::is_move_constructible<std::decay_t<Accum>>::value and
    std::is_copy_constructible<std::decay_t<Accum>>::value, int> = 0>
  constexpr decltype(auto) fold(const BinaryFunction& b, Accum&& accum, Arg&& arg)
#endif
  {
    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (std::is_same_v<BinaryFunction, std::plus> and (constant_matrix<Arg> or diagonal_matrix<Arg>))
    {
      if constexpr (zero_matrix<Arg>)
        return Scalar(0);
      else if constexpr (constant_matrix<Arg>)
        return Scalar(constant_coefficient_v<Arg> * row_count(arg) * column_count(arg));
      else if constexpr (constant_diagonal_matrix<Arg>)
        return Scalar(constant_diagonal_coefficient_v<Arg> * row_count(arg));
      else
      {
        static_assert(diagonal_matrix<Arg>);
        return OpenKalman::fold(b, std::forward<Accum>(accum), diagonal_of(std::forward<Arg>(arg)));
      }
    }
    else if constexpr (std::is_same_v<BinaryFunction, std::multiplies> and
      (constant_matrix<Arg> or triangular_matrix<Arg>))
    {
      if constexpr (zero_matrix<Arg> or triangular_matrix<Arg>)
        return Scalar(0);
      else
      {
        static_assert(constant_matrix<Arg>);
        if constexpr (dynamic_shape<Arg>)
          return Scalar(std::pow(constant_coefficient_v<Arg>, row_count(arg) * column_count(arg)));
        else
          return internal::constexpr_pow(constant_coefficient_v<Arg>, row_extent_of_v<Arg> * column_extent_of_v<Arg>);
      }
    }
    else
    {
      return interface::ElementWiseOperations<std::decay_t<Arg>>::template fold<order>(
        b, std::forward<Accum>(accum), std::forward<Arg>(arg));
    }
  }


  // ========================== //
  //  Linear algebra functions  //
  // ========================== //

  /**
   * \brief Take the conjugate of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) conjugate(Arg&& arg) noexcept
  {
    if constexpr (not complex_number<scalar_type_of_t<Arg>> or zero_matrix<Arg> or identity_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return std::forward<Arg>(arg);
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::conjugate(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the transpose of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) transpose(Arg&& arg) noexcept
  {
    if constexpr (diagonal_matrix<Arg> or (self_adjoint_matrix<Arg> and not complex_number<scalar_type_of_t<Arg>>) or
      (constant_matrix<Arg> and not dynamic_shape<Arg> and row_extent_of_v<Arg> == column_extent_of_v<Arg>))
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::transpose(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the adjoint of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg>
  constexpr decltype(auto) adjoint(Arg&& arg) noexcept
  {
    if constexpr (self_adjoint_matrix<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (zero_matrix<Arg> or not complex_number<scalar_type_of_t<Arg>>)
    {
      return transpose(std::forward<Arg>(arg));
    }
    else if constexpr (constant_matrix<Arg>)
    {
      if constexpr (std::imag(constant_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not dynamic_shape<Arg> and row_extent_of_v<Arg> == column_extent_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (std::imag(constant_diagonal_coefficient_v<Arg>) == 0)
        return transpose(std::forward<Arg>(arg));
      else if constexpr (not dynamic_shape<Arg> and row_extent_of_v<Arg> == column_extent_of_v<Arg>)
        return conjugate(std::forward<Arg>(arg));
      else
        return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
    else
    {
      return interface::LinearAlgebra<std::decay_t<Arg>>::adjoint(std::forward<Arg>(arg));
    }
  }


  /**
   * \brief Take the determinant of a matrix
   * \tparam Arg The matrix
   */
#ifdef __cpp_concepts
  template<typename Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<
    native_eigen_general<Arg> and (dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto determinant(Arg&& arg)
  {
    if constexpr (dynamic_shape<Arg>) if (row_count(arg) != column_count(arg))
      throw std::domain_error {
        "In determinant, rows of arg (" + std::to_string(row_count(arg)) + ") do not match columns of arg (" +
        std::to_string(column_count(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(1);
    }
    else if constexpr (zero_matrix<Arg> or (constant_matrix<Arg> and not one_by_one_matrix<Arg>))
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return constant_coefficient_v<Arg>; //< One-by-one case. General case is handled above.
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (dynamic_rows<Arg>)
        return std::pow(constant_diagonal_coefficient_v<Arg>, row_count(arg));
      else
        return OpenKalman::internal::constexpr_pow(constant_diagonal_coefficient_v<Arg>, row_extent_of_v<Arg>);
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::determinant(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      return r;
    }
  }


#ifdef __cpp_concepts
  /**
   * \brief Take the trace of a matrix
   * \tparam Arg The matrix
   */
  template<typename Arg> requires dynamic_shape<Arg> or square_matrix<Arg>
#else
  template<typename Arg, std::enable_if_t<(dynamic_shape<Arg> or square_matrix<Arg>), int> = 0>
#endif
  constexpr auto trace(Arg&& arg)
  {
    if constexpr (dynamic_shape<Arg>) if (row_count(arg) != column_count(arg))
      throw std::domain_error {
        "In trace, rows of arg (" + std::to_string(row_count(arg)) + ") do not match columns of arg (" +
        std::to_string(column_count(arg)) + ")"};

    using Scalar = scalar_type_of_t<Arg>;

    if constexpr (identity_matrix<Arg>)
    {
      return Scalar(row_count(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return Scalar(0);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      return Scalar(constant_coefficient_v<Arg> * row_count(arg));
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      return Scalar(constant_diagonal_coefficient_v<Arg> * row_count(arg));
    }
    else if constexpr (one_by_one_matrix<Arg> and element_gettable<Arg, std::size_t, std::size_t>)
    {
      return get_element(arg, std::size_t(0), std::size_t(0));
    }
    else
    {
      auto r = interface::LinearAlgebra<std::decay_t<Arg>>::trace(std::forward<Arg>(arg));
      static_assert(std::is_convertible_v<decltype(r), const scalar_type_of_t<Arg>>);
      return r;
    }
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a self-adjoint matrix.
   * \details If A is not hermitian, the result will modify only the specified storage triangle. The contents of the
   * other elements outside the specified storage triangle are undefined.
   * - The update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in hermitian form.
   */
#ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_extent_of_v<U> == row_extent_of_v<A>) and
    (dynamic_shape<A> or square_matrix<A>)
#else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    native_eigen_general<A> and native_eigen_general<U> and std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or
      row_extent_of<std::decay_t<U>>::value == row_extent_of<std::decay_t<A>>::value) and
    (dynamic_shape<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (row_count(a) != row_count(u))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(row_count(a)) + ") do not match rows of u (" +
        std::to_string(row_count(u)) + ")"};

    if constexpr (dynamic_shape<A>) if (row_count(a) != column_count(a))
      throw std::domain_error {
        "In rank_update_self_adjoint, rows of a (" + std::to_string(row_count(a)) + ") do not match columns of a (" +
        std::to_string(column_count(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<U> or dynamic_rows<A> or row_extent_of_v<U> == row_extent_of_v<A>) and
    (dynamic_shape<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    native_eigen_general<A> and native_eigen_general<U> and std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<U> or dynamic_rows<A> or row_extent_of<U>::value == row_extent_of<A>::value) and
    (dynamic_shape<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update_self_adjoint(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (self_adjoint_matrix<A>)
      return rank_update_self_adjoint<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (triangular_matrix<A>)
      return rank_update_self_adjoint<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a matrix, treating it as a triangular matrix.
   * \details If A is not a triangular matrix, the result will modify only the specified triangle. The contents of
   * other elements outside the specified triangle are undefined.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam t Whether to use the upper triangle elements (TriangleType::upper), lower triangle elements
   * (TriangleType::lower) or diagonal elements (TriangleType::diagonal).
   * \tparam A The matrix to be rank updated.
   * \tparam U The update vector or matrix.
   * \returns an updated native, writable matrix in triangular (or diagonal) form.
   */
# ifdef __cpp_concepts
  template<TriangleType t, typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_extent_of_v<A> == row_extent_of_v<U>) and
    (dynamic_shape<A> or square_matrix<A>)
# else
  template<TriangleType t, typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (t != TriangleType::lower or lower_triangular_matrix<A> or not upper_triangular_matrix<A>) and
    (t != TriangleType::upper or upper_triangular_matrix<A> or not lower_triangular_matrix<A>) and
    (dynamic_rows<A> or dynamic_rows<U> or row_extent_of<A>::value == row_extent_of<U>::value) and
    (dynamic_shape<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (row_count(a) != row_count(u))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(row_count(a)) + ") do not match rows of u (" +
        std::to_string(row_count(u)) + ")"};

    if constexpr (dynamic_shape<A>) if (row_count(a) != column_count(a))
      throw std::domain_error {
        "In rank_update_triangular, rows of a (" + std::to_string(row_count(a)) + ") do not match columns of a (" +
        std::to_string(column_count(a)) + ")"};

    if constexpr (zero_matrix<U>)
    {
      return std::forward<A>(a);
    }
    // \todo Generalize zero and diagonal cases from eigen and special-matrix
    else
    {
      using Trait = interface::LinearAlgebra<std::decay_t<A>>;
      return Trait::template rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
  }


  /**
   * \overload
   * \brief The triangle type is derived from A, or is TriangleType::lower by default.
   */
# ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (dynamic_rows<A> or dynamic_rows<U> or row_extent_of_v<A> == row_extent_of_v<U>) and
    (dynamic_shape<A> or square_matrix<A>)
# else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (dynamic_rows<A> or dynamic_rows<U> or row_extent_of<A>::value == row_extent_of<U>::value) and
    (dynamic_shape<A> or square_matrix<A>), int> = 0>
# endif
  inline decltype(auto)
  rank_update_triangular(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (triangular_matrix<A>)
      return rank_update_triangular<triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else if constexpr (self_adjoint_matrix<A>)
      return rank_update_triangular<self_adjoint_triangle_type_of_v<A>>(std::forward<A>(a), std::forward<U>(u), alpha);
    else
      return rank_update_triangular<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
  }


  /**
   * \brief Do a rank update on a hermitian, triangular, or one-by-one matrix.
   * \details
   * - If A is hermitian and non-diagonal, then the update is A += αUU<sup>*</sup>, returning the updated hermitian A.
   * - If A is lower-triangular, diagonal, or one-by-one, the update is AA<sup>*</sup> += αUU<sup>*</sup>,
   * returning the updated A.
   * - If A is upper-triangular, the update is A<sup>*</sup>A += αUU<sup>*</sup>, returning the updated A.
   * - If A is an lvalue reference and is writable, it will be updated in place and the return value will be an
   * lvalue reference to the same, updated A. Otherwise, the function returns a new matrix.
   * \tparam A A matrix to be updated, which must be hermitian (known at compile time),
   * triangular (known at compile time), or one-by-one (determinable at runtime) .
   * \tparam U A matrix or column vector with a number of rows that matches the size of A.
   * \param alpha A scalar multiplication factor.
   */
#ifdef __cpp_concepts
  template<typename A, typename U, std::convertible_to<const scalar_type_of_t<A>> Alpha> requires
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and dynamic_shape<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and dynamic_shape<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_extent_of_v<U> == row_extent_of_v<A>) and
    (dynamic_shape<A> or square_matrix<A>)
#else
  template<typename A, typename U, typename Alpha, std::enable_if_t<
    std::is_convertible_v<Alpha, const scalar_type_of_t<A>> and
    (triangular_matrix<A> or self_adjoint_matrix<A> or (zero_matrix<A> and dynamic_shape<A>) or
      (constant_matrix<A> and not complex_number<scalar_type_of<A>> and dynamic_shape<A>) or
      (dynamic_rows<A> and dynamic_columns<A>) or (dynamic_rows<A> and column_vector<A>) or
      (dynamic_columns<A> and row_vector<A>)) and
    (dynamic_rows<U> or dynamic_rows<A> or row_extent_of<U>::value == row_extent_of<A>::value) and
    (dynamic_shape<A> or square_matrix<A>), int> = 0>
#endif
  inline decltype(auto)
  rank_update(A&& a, U&& u, Alpha alpha = 1)
  {
    if constexpr (dynamic_rows<A> or dynamic_rows<U>) if (row_count(a) != row_count(u))
      throw std::domain_error {
        "In rank_update, rows of a (" + std::to_string(row_count(a)) + ") do not match rows of u (" +
        std::to_string(row_count(u)) + ")"};

    if constexpr (triangular_matrix<A>)
    {
      constexpr TriangleType t = triangle_type_of_v<A>;
      return rank_update_triangular<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (zero_matrix<A> and dynamic_shape<A>)
    {
      return rank_update_triangular<TriangleType::diagonal>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (self_adjoint_matrix<A>)
    {
      constexpr TriangleType t = self_adjoint_triangle_type_of_v<A>;
      return rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else if constexpr (constant_matrix<A> and not complex_number<scalar_type_of<A>> and dynamic_shape<A>)
    {
      return rank_update_self_adjoint<TriangleType::lower>(std::forward<A>(a), std::forward<U>(u), alpha);
    }
    else
    {
      if constexpr (dynamic_shape<A>)
        if ((dynamic_rows<A> and row_count(a) != 1) or (dynamic_columns<A> and column_count(a) != 1))
          throw std::domain_error {
            "Non hermitian, non-triangular argument to rank_update expected to be one-by-one, but instead it has " +
            std::to_string(row_count(a)) + " rows and " + std::to_string(column_count(a)) + " columns"};

      auto e = std::sqrt(trace(a) * trace(a.conjugate()) + alpha * trace(u * adjoint(u)));

      if constexpr (std::is_lvalue_reference_v<A> and not std::is_const_v<std::remove_reference_t<A>> and writable<A>)
      {
        if constexpr (element_settable<A, std::size_t, std::size_t>)
        {
          set_element(a, e, 0, 0);
        }
        else
        {
          a = MatrixTraits<A>::make(e);
        }
        return std::forward<A>(a);
      }
      else
      {
        return MatrixTraits<A>::make(e);
      }
    }
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
    // ------------------------ //
    //  to_covariance_nestable  //
    // ------------------------ //

    /**
     * \overload
     * \internal
     * \brief Convert a \ref covariance_nestable matrix or \ref typed_matrix_nestable to a \ref covariance_nestable.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_extent_of_v<Arg> == row_extent_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and
      (covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_extent_of<Arg>::value == row_extent_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \internal
     * \brief Convert \ref covariance or \ref typed_matrix to a \ref covariance_nestable of type T.
     * \tparam T \ref covariance_nestable to which Arg is to be converted.
     * \tparam Arg A \ref covariance or \ref typed_matrix.
     * \return A \ref covariance_nestable of type T.
     */
#ifdef __cpp_concepts
    template<covariance_nestable T, typename Arg> requires
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_extent_of_v<Arg> == row_extent_of_v<T>) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)
#else
    template<typename T, typename Arg, typename = void, typename = std::enable_if_t<
      (not std::is_same_v<T, Arg>) and covariance_nestable<T> and (not std::is_void_v<Arg>) and
      (covariance<Arg> or (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))) and
      (row_extent_of<Arg>::value == row_extent_of<T>::value) and
      (not zero_matrix<T> or zero_matrix<Arg>) and (not identity_matrix<T> or identity_matrix<Arg>) and
      (not diagonal_matrix<T> or diagonal_matrix<Arg> or column_vector<Arg>)>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return The result of converting Arg to a \ref covariance_nestable.
     */
#ifdef __cpp_concepts
    template<typename Arg>
    requires covariance_nestable<Arg> or (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = std::enable_if_t<covariance_nestable<Arg> or
        (typed_matrix_nestable<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;


    /**
     * \overload
     * \internal
     * /return A \ref triangular_matrix if Arg is a \ref triangular_covariance or otherwise a \ref self_adjoint_matrix.
     */
#ifdef __cpp_concepts
    template<typename Arg> requires covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))
#else
    template<typename Arg, typename = void, typename = std::enable_if_t<covariance<Arg> or
      (typed_matrix<Arg> and (square_matrix<Arg> or column_vector<Arg>))>>
#endif
    constexpr decltype(auto)
    to_covariance_nestable(Arg&&) noexcept;

  } // namespace internal


} // namespace OpenKalman

#endif //OPENKALMAN_FUNCTIONS_HPP
