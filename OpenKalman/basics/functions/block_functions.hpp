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
    constexpr void check_block_limit(const Arg& arg, const Limits&...)
    {
      if constexpr (((limit_ix >= std::tuple_size_v<Limits>) or ...)) return;
      else if constexpr (((static_index_value<std::tuple_element_t<limit_ix, Limits>>) and ...) and not dynamic_dimension<Arg, index>)
      {
        constexpr std::size_t block_limit = (std::size_t {std::tuple_element_t<limit_ix, Limits>{}} + ... + 0_uz);
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
      std::index_sequence<limit_ix...>, std::index_sequence<indices...>, const Arg& arg, const Limits&...limits)
    {
      (check_block_limit<limit_ix, indices>(arg, limits...), ...);
    }


    template<bool is_begin, std::size_t arg_ix, typename Arg>
    constexpr auto get_block_limits(const Arg& arg)
    {
      if constexpr (is_begin) return std::integral_constant<std::size_t, 0>{};
      else if constexpr (dynamic_dimension<Arg, arg_ix>) return get_index_dimension_of<arg_ix>(arg);
      else return std::integral_constant<std::size_t, index_dimension_of_v<Arg, arg_ix>>{};
    }


    template<bool is_begin, std::size_t arg_ix, std::size_t index, std::size_t...indices, typename Arg, typename Limit, typename...Limits>
    constexpr auto get_block_limits(const Arg& arg, const Limit& limit, const Limits&...limits)
    {
      if constexpr (arg_ix == index)
      {
        static_assert(((index != indices) and ...), "Duplicate index parameters are not allowed in block function.");
        return limit;
      }
      else
      {
        return get_block_limits<is_begin, arg_ix, indices...>(arg, limits...);
      }
    }


    template<bool is_begin, std::size_t...indices, typename Arg, typename Limit_tup, std::size_t...arg_ix, std::size_t...limits_ix>
    constexpr auto expand_block_limits(std::index_sequence<arg_ix...>, std::index_sequence<limits_ix...>, const Arg& arg, const Limit_tup& limit_tup)
    {
      return std::tuple {get_block_limits<is_begin, arg_ix, indices...>(arg, std::get<limits_ix>(limit_tup)...)...};
    }


    template<typename Arg, typename...Begin, typename...Size>
    constexpr auto block_impl(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size)
    {
      // \todo Extract the correct \ref vector_space_descriptor from Arg.
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
        return interface::library_interface<std::decay_t<Arg>>::get_block(std::forward<Arg>(arg), begin, size);
      }
    }

  } // namespace detail


  /**
   * \overload
   * \brief Extract a block from a matrix or tensor.
   * \details If indices are specified, only those indices will be subsetted. Otherwise, the Begin and Size parameters
   * are taken in index order. Any omitting trailing indices (for which there are no Begin or Size parameters) are included whole.
   * \tparam indices The index or indices of the particular dimensions to be specified, in any order (optional).
   * \param arg The indexible object from which a block is to be taken.
   * \param begin A tuple corresponding to each of indices, each element specifying the beginning \ref index_value.
   * If indices are not specified, the tuple proceeds in normal index order.
   * \param size A tuple corresponding to each of indices, each element specifying the dimensions of the extracted block.
   * If indices are not specified, the tuple proceeds in normal index order.
   * \todo Add a static check to ensure that the returned block has the expected vector space descriptors
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Begin, index_value...Size> requires
    (sizeof...(Begin) == sizeof...(Size)) and
    (sizeof...(indices) == 0 or ((has_uniform_dimension_type<vector_space_descriptor_of_t<Arg, indices>> or
      (static_index_value<Begin> and static_index_value<Size>)) and ...))
  constexpr indexible decltype(auto)
#else
  template<std::size_t...indices, typename Arg, typename...Begin, typename...Size, std::enable_if_t<
    indexible<Arg> and (index_value<Begin> and ...) and (index_value<Size> and ...) and
    (sizeof...(Begin) == sizeof...(Size)) and (sizeof...(indices) == 0 or sizeof...(indices) == sizeof...(Begin)), int> = 0>
  constexpr decltype(auto)
#endif
  get_block(Arg&& arg, const std::tuple<Begin...>& begin, const std::tuple<Size...>& size)
  {
    if constexpr (sizeof...(Begin) == 0)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      std::index_sequence_for<Begin...> begin_seq;
      std::conditional_t<sizeof...(indices) == 0, decltype(begin_seq), std::index_sequence<indices...>> indices_seq;
      detail::check_block_limits(begin_seq, indices_seq, arg, begin);
      detail::check_block_limits(begin_seq, indices_seq, arg, begin, size);

      if constexpr (sizeof...(indices) == 0)
      {
        return detail::block_impl(std::forward<Arg>(arg), begin, size);
      }
      else
      {
        auto arg_ix_seq = std::make_index_sequence<index_count_v<Arg>>{};
        return detail::block_impl(std::forward<Arg>(arg),
          detail::expand_block_limits<true, indices...>(arg_ix_seq, begin_seq, arg, begin),
          detail::expand_block_limits<false, indices...>(arg_ix_seq, begin_seq, arg, size));
      }
    }
  }


  // =========== //
  //  set_block  //
  // =========== //

  /**
   * \brief Extract a block from a matrix or tensor.
   * \param arg The indexible object in which the block is to be set.
   * \param block The block to be set.
   * \param begin A tuple specifying, for each index of Arg in order, the beginning \ref index_value.
   * \param size A tuple specifying, for each index of Arg in order, the dimensions of the extracted block.
   * \return arg as modified
   */
#ifdef __cpp_concepts
  template<writable Arg, indexible Block, index_value...Begin> requires (sizeof...(Begin) >= index_count_v<Arg>)
#else
  template<typename Arg, typename Block, typename...Begin, std::enable_if_t<writable<Arg> and indexible<Block> and
    (index_value<Begin> and ...) and (sizeof...(Begin) >= index_count<Arg>::value), int> = 0>
#endif
  constexpr Arg&&
  set_block(Arg&& arg, Block&& block, const Begin&...begin)
  {
    std::index_sequence_for<Begin...> begin_seq;
    detail::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...});
    detail::check_block_limits(begin_seq, begin_seq, arg, std::tuple{begin...},
      std::apply([](auto&&...a){
        return std::tuple{[](auto&& a){
          if constexpr (fixed_vector_space_descriptor<decltype(a)>)
            return std::integral_constant<std::size_t, dimension_size_of_v<decltype(a)>> {};
          else
            return get_dimension_size_of(std::forward<decltype(a)>(a));
        }(std::forward<decltype(a)>(a))...};
      }, get_all_dimensions_of(block)));

    interface::library_interface<std::decay_t<Arg>>::set_block(arg, std::forward<Block>(block), begin...);
    return std::forward<Arg>(arg);
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
   * If no indices are listed, the argument will be returned unchanged.
   * \param ixs The index value corresponding to each of the <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return A sub-array of the argument
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, indexible Arg, index_value...Ixs> requires (sizeof...(indices) == sizeof...(Ixs))
#else
  template<std::size_t...indices, typename Arg, typename...Ixs, std::enable_if_t<
    indexible<Arg> and (index_value<Ixs> and ...) and (sizeof...(indices) == sizeof...(Ixs)), int> = 0>
#endif
  constexpr decltype(auto)
  get_chip(Arg&& arg, Ixs...ixs)
  {
    if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
    else return get_block<indices...>(
      std::forward<Arg>(arg),
      std::tuple{ixs...}, // begin points
      std::tuple{(std::integral_constant<decltype(indices), 1> {})...}); // block sizes (always 1 in each collapsed dimension)
  }


  /**
   * \brief Extract one row from a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions. In effect, this function
   * collapses the row (0) index.
   * \param arg The matrix or other tensor from which the row is to be extracted
   * \param ix The index of the row
   */
#ifdef __cpp_concepts
  constexpr vector<1> decltype(auto)
  get_row(indexible auto&& arg, index_value auto ix)
#else
  template<typename Arg, typename Ix, std::enable_if_t<indexible<Arg> and index_value<Ix>, int> = 0>
  constexpr decltype(auto)
  get_row(Arg&& arg, Ix ix)
#endif
  {
    if constexpr (static_index_value<decltype(ix)> and not dynamic_dimension<decltype(arg), 0>)
      static_assert(ix < index_dimension_of_v<decltype(arg), 0>, "get_row: index must be in range");
    return get_chip<0>(std::forward<decltype(arg)>(arg), ix);
  }


  /**
   * \brief Extract one column from a matrix or other tensor.
   * \details If the tensor order is greater than 2, the column will have multiple dimensions. In effect, this function
   * collapses the column (1) index.
   * \tparam Arg The matrix or other tensor from which the column is to be extracted
   * \param ix The index of the column
   */
#ifdef __cpp_concepts
  constexpr vector<0> decltype(auto)
  get_column(indexible auto&& arg, index_value auto ix)
#else
  template<typename Arg, typename Ix, std::enable_if_t<indexible<Arg> and index_value<Ix>, int> = 0>
  constexpr decltype(auto)
  get_column(Arg&& arg, Ix ix)
#endif
  {
    if constexpr (static_index_value<decltype(ix)> and not dynamic_dimension<decltype(arg), 1>)
      static_assert(ix < index_dimension_of_v<decltype(arg), 1>, "get_column: index must be in range");
    return get_chip<1>(std::forward<decltype(arg)>(arg), ix);
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
    constexpr Arg& set_chip_impl(Arg&& arg, Chip&& chip, std::index_sequence<all_indices...>, Is...is)
    {
      return set_block(std::forward<Arg>(arg), std::forward<Chip>(chip), chip_index_match<all_indices, indices...>(is...)...);
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
   * \param arg The indexible object in which the chip is to be set.
   * \param chip The chip to be set. It must be a chip, meaning that the dimension is 1 for each of <code>indices</code>.
   * \param is The index value(s) corresponding to <code>indices</code>, in the same order. The values
   * may be positive \ref std::integral types or a positive \ref std::integral_constant.
   * \return arg as modified
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, writable Arg, indexible Chip, index_value...Is> requires
    (sizeof...(indices) == sizeof...(Is))
#else
  template<std::size_t...indices, typename Arg, typename Chip, typename...Is, std::enable_if_t<
    writable<Arg> and indexible<Chip> and (index_value<Is> and ...) and (sizeof...(indices) == sizeof...(Is)), int> = 0>
#endif
  constexpr Arg&&
  set_chip(Arg&& arg, Chip&& chip, Is...is)
  {
    ([](const auto& chip){
      if constexpr (not dynamic_dimension<Chip, indices>)
        static_assert(dimension_size_of_index_is<Chip, indices, 1>, "Argument to set_chip is not 1D in at least one of the specified indices.");
      else if (get_index_dimension_of<indices>(chip) != 1)
        throw std::invalid_argument {"Argument to set_chip must be 1D in each of the specified indices."};
    }(chip),...);

    return detail::set_chip_impl<indices...>(std::forward<Arg>(arg), std::forward<Chip>(chip),
      std::make_index_sequence<index_count_v<Arg>> {}, is...);
  }


  /**
   * \brief Set one row within a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions.
   * \param arg The indexible object in which the row is to be set.
   * \param row The row to be set.
   * \param i The type of the index of the row, which is an \index_value
   */
#ifdef __cpp_concepts
  template<writable Arg, vector<0, Likelihood::maybe> Column, index_value I> requires
    maybe_equivalent_to<vector_space_descriptor_of_t<Arg, 0>, vector_space_descriptor_of_t<Column, 0>>
#else
  template<typename Arg, typename Column, typename I, std::enable_if_t<writable<Arg> and
    vector<Column, 0, Likelihood::maybe> and index_value<I> and
    maybe_equivalent_to<vector_space_descriptor_of_t<Arg, 0>, vector_space_descriptor_of_t<Column, 0>>, int> = 0>
#endif
  constexpr Arg&&
  set_column(Arg&& arg, Column&& column, I i)
  {
    if constexpr (static_index_value<I> and not dynamic_dimension<Arg, 1>)
      static_assert(i < index_dimension_of_v<Arg, 1>, "set_column: index must be in range");
    if constexpr (not dynamic_dimension<Arg, 0> and not dynamic_dimension<Column, 0>)
      static_assert(index_dimension_of_v<Arg, 0> == index_dimension_of_v<Column, 0>, "set_column: column dimension must match argument row dimension");
    return set_chip<1>(std::forward<Arg>(arg), std::forward<Column>(column), i);
  }


  /**
   * \brief Set one row within a matrix or other tensor.
   * \details If the tensor order is greater than 2, the row will have multiple dimensions.
   * \param arg The indexible object in which the row is to be set.
   * \param row The row to be set.
   * \param i The type of the index of the row, which is an \index_value
   */
#ifdef __cpp_concepts
  template<writable Arg, vector<1, Likelihood::maybe> Row, index_value I> requires
    maybe_equivalent_to<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Row, 1>>
#else
  template<typename Arg, typename Row, typename I, std::enable_if_t<writable<Arg> and
    vector<Row, 1, Likelihood::maybe> and index_value<I> and
    maybe_equivalent_to<vector_space_descriptor_of_t<Arg, 1>, vector_space_descriptor_of_t<Row, 1>>, int> = 0>
#endif
  constexpr Arg&&
  set_row(Arg&& arg, Row&& row, I i)
  {
    if constexpr (static_index_value<I> and not dynamic_dimension<Arg, 0>)
      static_assert(i < index_dimension_of_v<Arg, 0>, "set_row: index must be in range");
    if constexpr (not dynamic_dimension<Arg, 1> and not dynamic_dimension<Row, 1>)
      static_assert(index_dimension_of_v<Arg, 1> == index_dimension_of_v<Row, 1>, "set_row: row dimension must match argument column dimension");
    return set_chip<0>(std::forward<Arg>(arg), std::forward<Row>(row), i);
  }


  // ==================== //
  //  Internal functions  //
  // ==================== //

  namespace internal
  {
    /**
     * \internal
     * \brief Set only a triangular (upper or lower) or diagonal part of a matrix by copying from another matrix.
     * \note This is optional.
     * \tparam t The TriangleType (upper, lower, or diagonal)
     * \param a The matrix or tensor to be set
     * \param b A matrix or tensor to be copied from, which may or may not be triangular
     */
#ifdef __cpp_concepts
    template<TriangleType t, square_matrix<Likelihood::maybe> A, square_matrix<Likelihood::maybe> B> requires
      maybe_has_same_shape_as<A, B> and (t != TriangleType::any) and
      (not triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<A, t, Likelihood::maybe> or t == TriangleType::diagonal) and
      (not triangular_matrix<B, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, t, Likelihood::maybe> or t == TriangleType::diagonal)
    constexpr maybe_has_same_shape_as<A> decltype(auto)
#else
    template<TriangleType t, typename A, typename B, std::enable_if_t<
      square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and
      maybe_has_same_shape_as<A, B> and (t != TriangleType::any) and
      (not triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<A, t, Likelihood::maybe> or t == TriangleType::diagonal) and
      (not triangular_matrix<B, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, t, Likelihood::maybe> or t == TriangleType::diagonal), int> = 0>
    constexpr decltype(auto)
#endif
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
      else if constexpr (interface::set_triangle_defined_for<std::decay_t<A>, t, A&&, B&&>)
      {
        return interface::library_interface<std::decay_t<A>>::template set_triangle<t>(std::forward<A>(a), std::forward<B>(b));
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

        return aw;
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
    constexpr maybe_has_same_shape_as<A> decltype(auto)
#else
    template<typename A, typename B, std::enable_if_t<
      square_matrix<A, Likelihood::maybe> and square_matrix<B, Likelihood::maybe> and maybe_has_same_shape_as<A, B> and
      (triangular_matrix<A, TriangleType::any, Likelihood::maybe> or triangular_matrix<B, TriangleType::any, Likelihood::maybe>) and
      (triangle_type_of<A>::value == TriangleType::any or triangle_type_of<B>::value == TriangleType::any or
        triangle_type_of<A, B>::value != TriangleType::any), int> = 0>
    constexpr decltype(auto)
#endif
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
