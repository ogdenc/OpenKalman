/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Functions relating to block subsets of math objects.
 */

#ifndef OPENKALMAN_BLOCK_FUNCTIONS_HPP
#define OPENKALMAN_BLOCK_FUNCTIONS_HPP

#include "functions.hpp"

namespace OpenKalman
{
  // =========== //
  //  get_block  //
  // =========== //

  namespace detail
  {
    template<std::size_t limit_ix, std::size_t index, typename Arg, typename...Limits>
    constexpr void check_block_limit(const Arg& arg, Limits...limits)
    {
      if constexpr (((static_index_value<std::tuple_element_t<limit_ix, Limits>>) and ...) and not dynamic_dimension<Arg, index>)
      {
        constexpr std::size_t block_limit = (static_index_value_of_v<std::tuple_element_t<limit_ix, Limits>> + ... + 0);
        static_assert(block_limit <= index_dimension_of_v<Arg, index>, "Block limits must be in range");
      }
      /*else // Not necessary: the matrix/tensor library should check runtime limits.
      {
        auto lim = (std::get<limit_ix>(limits) + ... + 0);
        auto max = get_index_dimension_of<index>(arg);
        if (lim < 0 or lim > max) throw std::out_of_range {"Block function limits are out of range for index " + std::to_string(index)};
      }*/
    }

    template<std::size_t...limit_ix, std::size_t...indices, typename Arg, typename...Limits>
    constexpr void check_block_limits(
      std::index_sequence<limit_ix...>, std::index_sequence<indices...>, const Arg& arg, Limits...limits)
    {
      (check_block_limit<limit_ix, std::get<limit_ix>(std::make_tuple(indices...))>(arg, limits...), ...);
    }


    template<typename Arg, typename...Begin, typename...Size>
    constexpr auto block_impl(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
    {
      // \todo Extract the correct index descriptors from Arg.
      if constexpr (zero_matrix<Arg>)
      {
        return std::apply([](auto...ds){ return make_zero_matrix_like<Arg>(ds...); }, size);
      }
      else if constexpr (constant_matrix<Arg>)
      {
        return std::apply(
          [](const auto& c, auto...ds){ return make_constant_matrix_like<Arg>(c, ds...); },
          std::tuple_cat(std::tuple{constant_coefficient{arg}}, size));
      }
      else
      {
        return interface::LibraryRoutines<std::decay_t<Arg>>::get_block(std::forward<Arg>(arg), begin, size);
      }
    }

  } // namespace detail


  /**
   * \brief Extract a block from a matrix or tensor.
   * \tparam Arg The indexible object from which a block is to be taken.
   * \param begin A tuple specifying, for each index of Arg in order, the beginning \ref index_value.
   * \param size A tuple specifying, for each index of Arg in order, the dimensions of the extracted block.
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_value...Begin, index_value...Size> requires
    (sizeof...(Begin) == max_indices_of_v<Arg>) and (sizeof...(Size) == max_indices_of_v<Arg>)
#else
  template<typename Arg, typename...Begin, typename...Size, std::enable_if_t<
    indexible<Arg> and (index_value<Begin> and ...) and (index_value<Size> and ...) and
    (sizeof...(Begin) == max_indices_of<Arg>::value and sizeof...(Size) == max_indices_of<Arg>::value), int> = 0>
#endif
  constexpr auto get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
  {
    std::make_index_sequence<max_indices_of_v<Arg>> seq;
    detail::check_block_limits(seq, seq, arg, begin);
    detail::check_block_limits(seq, seq, arg, begin, size);

    return detail::block_impl(std::forward<Arg>(arg), begin, size);
  }


  namespace detail
  {
    template<bool is_begin, std::size_t arg_ix, typename Arg>
    constexpr auto get_block_limits(const Arg& arg)
    {
      if constexpr (is_begin) return std::integral_constant<std::size_t, 0>{};
      else if constexpr (dynamic_dimension<Arg, arg_ix>) return get_index_dimension_of<arg_ix>(arg);
      else return std::integral_constant<std::size_t, index_dimension_of_v<Arg, arg_ix>>{};
    }

    template<bool is_begin, std::size_t arg_ix, std::size_t index, std::size_t...indices, typename Arg, typename Limit, typename...Limits>
    constexpr auto get_block_limits(const Arg& arg, Limit limit, Limits...limits)
    {
      if constexpr (arg_ix == index)
      {
        static_assert(((index != indices) and ...), "No duplicate index parameters to block function.");
        return limit;
      }
      else
      {
        return get_block_limits<is_begin, arg_ix, indices...>(arg, limits...);
      }
    }

    template<bool is_begin, std::size_t...indices, typename Arg, typename Limit_tup, std::size_t...arg_ix, std::size_t...limits_ix>
    constexpr auto expand_block_limits(const Arg& arg, Limit_tup limit_tup, std::index_sequence<arg_ix...>, std::index_sequence<limits_ix...>)
    {
      return std::tuple {get_block_limits<is_begin, arg_ix, indices...>(arg, std::get<limits_ix>(limit_tup)...)...};
    }

  } // namespace detail


  /**
   * \overload
   * \brief Extract a block from a matrix or tensor, but only subsetting the specified indices.
   * \details If an index is not specified, the entire index range will be used.
   * \tparam indices The index or indices of the particular dimensions to be specified, in any order.
   * \tparam Arg The indexible object from which a block is to be taken.
   * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
   * \param size A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Begin, index_value...Size> requires
    (sizeof...(indices) > 0) and ((indices < max_indices_of_v<Arg>) and ...) and
    (sizeof...(indices) == sizeof...(Begin)) and (sizeof...(indices) == sizeof...(Size))
#else
  template<std::size_t...indices, typename Arg, typename...Begin, typename...Size, std::enable_if_t<
    (sizeof...(indices) > 0) and ((indices < max_indices_of_v<Arg>) and ...) and
    indexible<Arg> and (index_value<Begin> and ...) and (index_value<Size> and ...) and
    (sizeof...(indices) == sizeof...(Begin) and sizeof...(indices) == sizeof...(Size)), int> = 0>
#endif
  constexpr auto get_block(Arg&& arg, std::tuple<Begin...> begin, std::tuple<Size...> size)
  {
    std::index_sequence<indices...> indices_seq;
    std::make_index_sequence<sizeof...(indices)> limits_ix_seq;

    detail::check_block_limits(limits_ix_seq, indices_seq, arg, begin);
    detail::check_block_limits(limits_ix_seq, indices_seq, arg, begin, size);

    auto arg_ix_seq = std::make_index_sequence<max_indices_of_v<Arg>>{};

    return detail::block_impl(std::forward<Arg>(arg),
      detail::expand_block_limits<true, indices...>(arg, begin, arg_ix_seq, limits_ix_seq),
      detail::expand_block_limits<false, indices...>(arg, size, arg_ix_seq, limits_ix_seq));
  }


  // =========== //
  //  set_block  //
  // =========== //

  /**
   * \brief Extract a block from a matrix or tensor.
   * \tparam Arg The indexible object in which the block is to be set.
   * \tparam Block The block to be set.
   * \param begin A tuple specifying, for each index of Arg in order, the beginning \ref index_value.
   * \param size A tuple specifying, for each index of Arg in order, the dimensions of the extracted block.
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Block, index_value...Begin> requires (sizeof...(Begin) == max_indices_of_v<Arg>)
#else
  template<typename Arg, typename Block, typename...Begin, std::enable_if_t<writable<Arg> and indexible<Block> and
    (index_value<Begin> and ...) and (sizeof...(Begin) == max_indices_of<Arg>::value), int> = 0>
#endif
  constexpr Arg& set_block(Arg& arg, Block&& block, Begin...begin)
  {
    auto seq = std::make_index_sequence<max_indices_of_v<Arg>>{};
    detail::check_block_limits(seq, seq, arg, std::tuple{begin...});
    detail::check_block_limits(seq, seq, arg, std::tuple{begin...},
      std::apply([](auto&&...a){
        return std::tuple{[](auto&& a){
          if constexpr (fixed_index_descriptor<decltype(a)>)
            return std::integral_constant<std::size_t, dimension_size_of_v<decltype(a)>> {};
          else
            return get_dimension_size_of(std::forward<decltype(a)>(a));
        }(std::forward<decltype(a)>(a))...};
      }, get_all_dimensions_of(block)));

    interface::LibraryRoutines<std::decay_t<Arg>>::set_block(arg, std::forward<Block>(block), begin...);
    return arg;
  }


  // ============================================================== //
  //  get_chip, set_chip, get_row, set_row, get_column, set_column  //
  // ============================================================== //

  /**
   * \brief Extract a sub-array having rank less than the rank of the input object.
   * \details A chip is a special type of "thin" slice of width 1 in one or more dimensions, and otherwise no
   * reduction in extents. For example, the result could be a row vector, a column vector, a matrix (e.g., if the
   * input object is a rank-3 or higher tensor), etc.
   * \tparam indices The index or indices of the dimension(s) to be collapsed to a single dimension.
   * For example, if the input object is a matrix, a value of {0} will result in a row vector, a value of {1} will
   * result in a column vector, and a value of {0, 1} will result in a one-dimensional vector.
   * If the input object is a rank-3 tensor, a value of {1, 2} will result in a row vector.
   * Omission of indices will return the argument unchanged.
   * \tparam Is The index value(s) corresponding to <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return A sub-array
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Is>
  requires (sizeof...(indices) > 0 and sizeof...(indices) == sizeof...(Is)) or
    (sizeof...(Is) == 0 and (dimension_size_of_index_is<Arg, indices, 1> and ...))
#else
  template<std::size_t...indices, typename Arg, typename...Is, std::enable_if_t<
    indexible<Arg> and (index_value<Is> and ...) and
    ((sizeof...(indices) > 0 and sizeof...(indices) == sizeof...(Is)) or
      (sizeof...(Is) == 0 and (dimension_size_of_index_is<Arg, indices, 1> and ...))), int> = 0>
#endif
  constexpr decltype(auto) get_chip(Arg&& arg, Is...is)
  {
    if constexpr (sizeof...(indices) > 0) return get_block<indices...>(std::forward<Arg>(arg),
      std::tuple{is...}, // begin points
      std::tuple{(std::integral_constant<decltype(indices), 1> {})...}); // sizes == 1 in each collapsed dimension
    else return std::forward<Arg>(arg);
  }


  /**
   * \brief Extract one row from a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions. In effect, this function
   * collapses the row (0) index.
   * \tparam Arg The matrix or other tensor from which the row is to be extracted
   * \tparam I The type of the index of the row, which is an \index_value
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_value I> requires dynamic_rows<Arg> or (not static_index_value<I>) or
    (static_index_value_of_v<I> < row_dimension_of_v<Arg>)
  constexpr vector<1> decltype(auto)
#else
  template<typename Arg, typename I, std::enable_if_t<indexible<Arg> and index_value<I> and
    (dynamic_rows<Arg> or not static_index_value<I> or (static_index_value_of<I>::value < row_dimension_of<Arg>::value)), int> = 0>
  constexpr decltype(auto)
#endif
  get_row(Arg&& arg, I i)
  {
    return get_chip<0>(std::forward<Arg>(arg), i);
  }


  /**
   * \brief Extract one column from a matrix or other tensor.
   * \details If the tensor order is greater than 2, the column will have multiple dimensions. In effect, this function
   * collapses the column (1) index.
   * \tparam Arg The matrix or other tensor from which the column is to be extracted
   * \tparam I The type of the index of the column, which is an \index_value
   */
#ifdef __cpp_concepts
  template<indexible Arg, index_value I> requires dynamic_columns<Arg> or (not static_index_value<I>) or
    (static_index_value_of_v<I> < column_dimension_of_v<Arg>)
  constexpr vector<0> decltype(auto)
#else
  template<typename Arg, typename I, std::enable_if_t<indexible<Arg> and index_value<I> and
    (dynamic_columns<Arg> or not static_index_value<I> or (static_index_value_of<I>::value < column_dimension_of<Arg>::value)), int> = 0>
  constexpr decltype(auto)
#endif
  get_column(Arg&& arg, I i)
  {
    return get_chip<1>(std::forward<Arg>(arg), i);
  }


  namespace detail
  {
    template<std::size_t>
    constexpr auto chip_index_match() { return std::integral_constant<std::size_t, 0> {}; }

    template<std::size_t ai, std::size_t index, std::size_t...indices, typename I, typename...Is>
    constexpr auto chip_index_match(I i, Is...is)
    {
      if constexpr (ai == index) return i;
      else return chip_index_match<ai, indices...>(is...);
    }

    template<std::size_t...indices, typename Arg, typename Chip, std::size_t...all_indices, typename...Is>
    constexpr auto& set_chip_impl(Arg& arg, Chip&& chip, std::index_sequence<all_indices...>, Is...is)
    {
      return set_block(arg, std::forward<Chip>(chip), chip_index_match<all_indices, indices...>(is...)...);
    }
  } // namespace detail


  /**
   * \brief Set a sub-array having rank less than the rank of the input object.
   * \details A chip is a special type of "thin" slice of width 1 in one or more dimensions, and otherwise no
   * reduction in extents. For example, the result could be a row vector, a column vector, a matrix (e.g., if the
   * input object is a rank-3 or higher tensor), etc.
   * \tparam indices The index or indices of the dimension(s) that have been collapsed to a single dimension.
   * For example, if the input object is a matrix, a value of {0} will result in a row vector and a value of {1} will
   * result in a column vector. If the input object is a rank-3 tensor, a value of {0, 1} will result in a matrix.
   * \tparam Arg The indexible object in which the chip is to be set.
   * \tparam Chip The chip to be set. It must be a chip, meaning that the dimension is 1 for each of <code>indices</code>.
   * \tparam Is The index value(s) corresponding to <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, writable Arg, indexible Chip, index_value...Is>
  requires ((sizeof...(indices) > 0 and sizeof...(indices) == sizeof...(Is)) or
    (sizeof...(Is) == 0 and (dimension_size_of_index_is<Arg, indices, 1> and ...)))
#else
  template<std::size_t...indices, typename Arg, typename Chip, typename...Is, std::enable_if_t<
    writable<Arg> and indexible<Chip> and (index_value<Is> and ...) and
    ((sizeof...(indices) > 0 and sizeof...(indices) == sizeof...(Is)) or
      (sizeof...(Is) == 0 and (dimension_size_of_index_is<Arg, indices, 1> and ...))), int> = 0>
#endif
  constexpr auto& set_chip(Arg& arg, Chip&& chip, Is...is)
  {
    ([](const auto& chip){
      if constexpr (not dynamic_dimension<Chip, indices>)
        static_assert(dimension_size_of_index_is<Chip, indices, 1>, "Argument to set_chip is not 1D in at least one of the specified indices.");
      else if (get_index_dimension_of<indices>(chip) != 1)
        throw std::invalid_argument {"Argument to set_chip must be 1D in each of the specified indices."};
    }(chip),...);

    return detail::set_chip_impl<indices...>(arg, std::forward<Chip>(chip),
      std::make_index_sequence<max_indices_of_v<Arg>> {}, is...);
  }


  /**
   * \brief Set one row within a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions.
   * \tparam Arg The indexible object in which the row is to be set.
   * \tparam Row The row to be set.
   * \tparam I The type of the index of the row, which is an \index_value
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Row, index_value I> requires dynamic_rows<Arg> or (not static_index_value<I>) or
    (static_index_value_of_v<I> < row_dimension_of_v<Arg>)
#else
  template<typename Arg, typename Row, typename I, std::enable_if_t<writable<Arg> and indexible<Row> and index_value<I> and
    (dynamic_rows<Arg> or not static_index_value<I> or (static_index_value_of<I>::value < row_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr auto&
  set_row(Arg& arg, Row&& row, I i)
  {
    return set_chip<0>(arg, std::forward<Row>(row), i);
  }


  /**
   * \brief Set one row within a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions.
   * \tparam Arg The indexible object in which the row is to be set.
   * \tparam Row The row to be set.
   * \tparam I The type of the index of the row, which is an \index_value
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Column, index_value I> requires dynamic_columns<Arg> or (not static_index_value<I>) or
    (static_index_value_of_v<I> < column_dimension_of_v<Arg>)
#else
  template<typename Arg, typename Column, typename I, std::enable_if_t<writable<Arg> and indexible<Column> and index_value<I> and
    (dynamic_columns<Arg> or not static_index_value<I> or (static_index_value_of<I>::value < column_dimension_of<Arg>::value)), int> = 0>
#endif
  constexpr auto&
  set_column(Arg& arg, Column&& column, I i)
  {
    return set_chip<1>(arg, std::forward<Column>(column), i);
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
#ifndef __cpp_concepts
    namespace detail
    {
      template<TriangleType t, typename A, typename B, typename = void>
      struct set_triangle_trait_exists : std::false_type {};

      template<TriangleType t, typename A, typename B>
      struct set_triangle_trait_exists<t, A, B, std::void_t<decltype(
          interface::LibraryRoutines<std::decay_t<A>>::template set_triangle<t>(std::declval<A>(), std::declval<B>()))>>
        : std::true_type {};
    }
#endif

    /**
     * \internal
     * \brief Set only a triangle (upper or lower) or diagonal taken from another matrix to a \ref writable matrix.
     * \note This is optional.
     * \tparam t The TriangleType (upper, lower, or diagonal)
     * \tparam A The matrix or tensor to be set
     * \tparam B A matrix or tensor to be copied from, which may or may not be triangular
     */
#ifdef __cpp_concepts
    template<TriangleType t, square_matrix<Likelihood::maybe> A, square_matrix<Likelihood::maybe> B> requires
      maybe_has_same_shape_as<A, B> and (t != TriangleType::any) and
      (not triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<A, t, Likelihood::maybe> or t == TriangleType::diagonal) and
      (not triangular_matrix<B, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, t, Likelihood::maybe> or t == TriangleType::diagonal)
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and
      maybe_has_same_shape_as<A, B> and (t != TriangleType::any) and
      (not triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<A, t, Likelihood::maybe> or t == TriangleType::diagonal) and
      (not triangular_matrix<B, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, t, Likelihood::maybe> or t == TriangleType::diagonal), int> = 0>
#endif
    constexpr decltype(auto)
    set_triangle(A&& a, B&& b)
    {
      if constexpr (diagonal_adapter<A>)
      {
        static_assert(t == TriangleType::diagonal);
        using N = decltype(nested_matrix(a));
        if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
        {
          nested_matrix(a) = diagonal_of(std::forward<B>(b));
          return std::forward<A>(a);
        }
        else
        {
          return to_diagonal(diagonal_of(std::forward<B>(b)));
        }
      }
      else if constexpr (triangular_adapter<A>)
      {
        using N = decltype(nested_matrix(a));
        if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
        {
          set_triangle<t>(nested_matrix(a), std::forward<B>(b));
          return std::forward<A>(a);
        }
        else
        {
          auto aw = make_dense_writable_matrix_from(nested_matrix(std::forward<A>(a)));
          set_triangle<t>(aw, std::forward<B>(b));
          return make_triangular_matrix<triangle_type_of_v<A>>(std::move(aw));
        }
      }
      else if constexpr (hermitian_adapter<A>)
      {
        using N = decltype(nested_matrix(a));
        if constexpr (writable<N> and std::is_lvalue_reference_v<N>)
        {
          if constexpr ((t == TriangleType::lower and hermitian_adapter<A, HermitianAdapterType::upper>) or
              (t == TriangleType::upper and hermitian_adapter<A, HermitianAdapterType::lower>))
            set_triangle<t>(adjoint(nested_matrix(a)), std::forward<B>(b));
          else
            set_triangle<t>(nested_matrix(a), std::forward<B>(b));
          return std::forward<A>(a);
        }
        else if constexpr (t == TriangleType::diagonal)
        {
          return make_hermitian_matrix<HermitianAdapterType::lower>(set_triangle<t>(nested_matrix(std::forward<A>(a)), std::forward<B>(b)));
        }
        else if constexpr (t == TriangleType::upper)
        {
          return make_hermitian_matrix<HermitianAdapterType::upper>(std::forward<B>(b));
        }
        else
        {
          return make_hermitian_matrix<HermitianAdapterType::lower>(std::forward<B>(b));
        }
      }
#ifdef __cpp_concepts
      else if constexpr (requires { interface::LibraryRoutines<std::decay_t<A>>::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b)); })
#else
      if constexpr (detail::set_triangle_trait_exists<t, A, B>::value)
#endif
      {
        return interface::LibraryRoutines<std::decay_t<A>>::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
      }
      else
      {
        decltype(auto) aw = make_dense_writable_matrix_from(std::forward<A>(a));

        if constexpr (t == TriangleType::upper)
        {
          for (int i = 0; i < get_index_dimension_of<0>(aw); i++)
          for (int j = i; j < get_index_dimension_of<1>(aw); j++)
            set_element(aw, get_element(b, i, i), i, i);
        }
        else if constexpr (t == TriangleType::lower)
        {
          for (int i = 0; i < get_index_dimension_of<0>(aw); i++)
          for (int j = 0; j < i; j++)
            set_element(aw, get_element(b, i, i), i, i);
        }
        else // t == TriangleType::diagonal
        {
          for (int i = 0; i < get_index_dimension_of<0>(aw); i++) set_element(aw, get_element(b, i, i), i, i);
        }
        return std::forward<decltype(aw)>(aw);
      }
    }


    /**
     * \overload
     * \internal
     * \brief Derives the TriangleType from the triangle types of the arguments.
     */
#ifdef __cpp_concepts
    template<square_matrix<Likelihood::maybe> A, square_matrix<Likelihood::maybe> B> requires maybe_has_same_shape_as<A, B> and
      (triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, TriangleType::any, Likelihood::maybe>) and
      (triangle_type_of_v<A> == TriangleType::any or triangle_type_of_v<B> == TriangleType::any or triangle_type_of_v<A, B> != TriangleType::any)
#else
    template<typename A, typename B, std::enable_if_t<
      square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and maybe_has_same_shape_as<A, B> and
      (triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, TriangleType::any, Likelihood::maybe>) and
      (triangle_type_of<A>::value == TriangleType::any or triangle_type_of<B>::value == TriangleType::any or
        triangle_type_of<A, B>::value != TriangleType::any), int> = 0>
#endif
    constexpr decltype(auto)
    set_triangle(A&& a, B&& b)
    {
      constexpr auto t =
        diagonal_matrix<A, Likelihood::maybe> or diagonal_matrix<B, Likelihood::maybe> ? TriangleType::diagonal :
        triangle_type_of_v<A, B> != TriangleType::any ? triangle_type_of_v<A, B> :
        triangle_type_of_v<A> != TriangleType::any ? triangle_type_of_v<A> : triangle_type_of_v<B>;
      return set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
    }

  } // namespace internal

} // namespace OpenKalman

#endif //OPENKALMAN_BLOCK_FUNCTIONS_HPP
