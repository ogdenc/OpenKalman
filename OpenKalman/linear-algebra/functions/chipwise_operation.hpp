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
 * \brief Chipwise n-ary operations.
 */

#ifndef OPENKALMAN_CHIPWISE_OPERATION_HPP
#define OPENKALMAN_CHIPWISE_OPERATION_HPP

namespace OpenKalman
{
  namespace detail
  {
    //-- chipwise_vector_space_descriptor_for --//

    template<std::size_t ix, typename Best_d>
    const Best_d& chipwise_vector_space_descriptor_for(const Best_d& best_d) { return best_d; }

    template<std::size_t ix, typename Best_d, typename Arg, typename...Args>
    decltype(auto) chipwise_vector_space_descriptor_for(const Best_d& best_d, const Arg& arg, const Args&...args)
    {
      auto d = get_vector_space_descriptor<ix>(arg);
      using D = decltype(d);
      if constexpr (fixed_pattern<Best_d>)
      {
        if constexpr (fixed_pattern<D>)
          static_assert(coordinate::size_of_v<D> == coordinate::size_of_v<Best_d>,
            "Arguments to chipwise_operation must have matching vector space descriptors.");
        else
          if (d != best_d) throw std::invalid_argument {"Arguments to chipwise_operation must have matching vector space descriptors."};
        return chipwise_vector_space_descriptor_for<ix>(best_d, args...);
      }
      else // dynamic_pattern<Best_d>
      {
        if (d != best_d) throw std::invalid_argument {"Arguments to chipwise_operation must have matching vector space descriptors."};
        if constexpr (fixed_pattern<D>)
          return chipwise_vector_space_descriptor_for<ix>(d, args...);
        else
          return chipwise_vector_space_descriptor_for<ix>(best_d, args...);
      }
    }

    //-- make_chipwise_default --//

    template<std::size_t...ix, typename Arg, typename...Args>
    auto make_chipwise_default(std::index_sequence<ix...>, const Arg& arg, const Args&...args)
    {
      return make_dense_object<Arg>(chipwise_vector_space_descriptor_for<ix>(get_vector_space_descriptor<ix>(arg), args...)...);
    }

    //-- chipwise_op_chip --//

    template<bool uses_indices, std::size_t...indices, std::size_t...indices_ix,
      typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    chipwise_op_chip(std::index_sequence<indices...>, std::index_sequence<indices_ix...>, const Ix_tup& ix_tup,
      M& m, const Op& op, Args&&...args)
    {
      if constexpr (uses_indices)
      {
        auto chip = op(get_chip<indices...>(std::forward<Args>(args), std::get<indices_ix>(ix_tup)...)..., std::get<indices_ix>(ix_tup)...);
        set_chip<indices...>(m, std::move(chip), std::get<indices_ix>(ix_tup)...);
      }
      else
      {
        auto chip = op(get_chip<indices...>(std::forward<Args>(args), std::get<indices_ix>(ix_tup)...)...);
        set_chip<indices...>(m, std::move(chip), std::get<indices_ix>(ix_tup)...);
      }
    }

    //-- chipwise_op --//

    template<bool uses_indices, typename Indices, typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    chipwise_op(Indices indices, const Ix_tup& ix_tup, M& m, const Op& op, Args&&...args)
    {
      constexpr auto num_indices = Indices::size();
      static_assert(std::tuple_size_v<Ix_tup> == num_indices);
      std::make_index_sequence<num_indices> indices_seq;
      chipwise_op_chip<uses_indices>(indices, indices_seq, ix_tup, m, op, std::forward<Args>(args)...);
    }

    template<bool uses_indices, std::size_t index, std::size_t...indices,
      typename Indices_seq, typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    chipwise_op(Indices_seq indices_seq, const Ix_tup& ix_tup, M& m, const Op& op, Args&&...args)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<index>(m); ++i)
        chipwise_op<uses_indices, indices...>(indices_seq, std::tuple_cat(ix_tup, std::tuple{i}), m, op, args...);
    }

  } // namespace detail


  /**
   * \brief Perform a chipwise n-ary operation (n&gt;0) on one or more indexible objects.
   * \details Given an indexible object of order <var>k</var>, a "chip" is a subset of that object, having order in
   * the range (0, <var>k</var>]. This function takes same-size chips from each of the arguments and
   * performs an operation returning a chip (of the same size), for every possible chip within the result.
   * \tparam indices The one-dimensional indices of the chip (optionally excluding any trailing 1D indices).
   * If omitted, the order of the chip is the same as that of the highest-order indexible argument.
   * Note: the list of indices must be non-repeating, or a compile-time assertion will fail.
   * \tparam Operation An n-ary operation (unary, binary, etc.) on n chips of the same size.
   * In addition to taking one or more chips as arguments, the operation may optionally take
   * <code>sizeof...(indices)</code> indices (in the same order as <code>indices</code>).
   * \tparam Args The arguments, which must be the same size.
   * \result An object of the same size as the highest-order argument. For example, a chipwise operation between a
   * 3&times;4 matrix and either a 3&times;1 row or 1&times;4 column vector is a 3&times;4 matrix.
   * (The vector is replicated vertically or horizontally, respectively, to fill the size of the matrix.)
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, typename Operation, indexible...Args> requires (sizeof...(Args) > 0)
#else
  template<std::size_t...indices, typename Operation, typename...Args, std::enable_if_t<
    (indexible<Args> and ...) and (sizeof...(Args) > 0), int> = 0>
#endif
  constexpr auto
  chipwise_operation(const Operation& operation, Args&&...args)
  {
    if constexpr (sizeof...(indices) > 0)
    {
      auto m = detail::make_chipwise_default(std::make_index_sequence<std::max({index_count_v<Args>...})>{}, args...);

      constexpr bool uses_indices = std::is_invocable_v<Operation,
        decltype(get_chip<indices...>(std::declval<Args>(), std::integral_constant<decltype(indices), 0>{}...))...,
        std::integral_constant<decltype(indices), 0>...>;

      std::index_sequence<indices...> indices_seq;
      detail::chipwise_op<uses_indices, indices...>(indices_seq, std::tuple{}, m, operation, std::forward<Args>(args)...);
      return m;
    }
    else
    {
      return operation(std::forward<Args>(args)...);
    }
  }


  namespace detail
  {
    template<std::size_t op_ix, typename OpResult>
    auto nullary_chipwise_vector_space_descriptor(const OpResult& op_result)
    {
      return get_vector_space_descriptor<op_ix>(op_result);
    }

    template<std::size_t op_ix, std::size_t index, std::size_t...indices, typename OpResult, typename I, typename...Is>
    auto nullary_chipwise_vector_space_descriptor(const OpResult& op_result, I i, Is...is)
    {
      if constexpr (op_ix == index) return get_vector_space_descriptor<op_ix>(op_result) * i;
      else return nullary_chipwise_vector_space_descriptor<op_ix, indices...>(op_result, is...);
    }

    template<std::size_t...indices, std::size_t...op_ix, typename OpResult, typename...Is>
    auto make_nullary_chipwise_default(std::index_sequence<op_ix...>, const OpResult& op_result, Is...is)
    {
      return make_dense_object<OpResult>(nullary_chipwise_vector_space_descriptor<op_ix, indices...>(op_result, is...)...);
    }


    template<bool uses_indices, std::size_t...indices, std::size_t...index_ixs,
      typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    nullary_chipwise_op_chip(std::index_sequence<indices...>, std::index_sequence<index_ixs...>, const Ix_tup& ix_tup,
      M& m, const Op& op)
    {
      [](M& m, const Op& op, auto...ix){
        if constexpr (uses_indices)
          set_chip<indices...>(m, op(ix...), ix...);
        else
          set_chip<indices...>(m, op(), ix...);
      }(m, op, std::get<index_ixs>(ix_tup)...);
    }

    template<bool uses_indices, bool first, typename All_index_seq, typename Ix_tup, typename M, typename Op>
    constexpr void
    nullary_chipwise_op(All_index_seq all_index_seq, const Ix_tup& ix_tup, M& m, const Op& op)
    {
      if constexpr (not first)
      {
        std::make_index_sequence<std::tuple_size_v<Ix_tup>> index_ix_seq;
        nullary_chipwise_op_chip<uses_indices>(all_index_seq, index_ix_seq, ix_tup, m, op);
      }
    }

    template<bool uses_indices, bool first, std::size_t index, std::size_t...indices, typename All_index_seq,
      typename Ix_tup, typename M, typename Op, typename I, typename...Is>
    constexpr void
    nullary_chipwise_op(All_index_seq all_index_seq, const Ix_tup& ix_tup, M& m, const Op& op, I i, Is...is)
    {
      if constexpr (first)
      {
        auto new_ix_tup = std::tuple_cat(ix_tup, std::tuple{std::integral_constant<std::size_t, 0> {}});
        nullary_chipwise_op<uses_indices, true, indices...>(all_index_seq, new_ix_tup, m, op, is...);
      }
      constexpr std::size_t begin = first ? 1 : 0;

      for (std::size_t j = begin; j < static_cast<std::size_t>(i); ++j)
      {
        auto new_ix_tup = std::tuple_cat(ix_tup, std::tuple{j});
        nullary_chipwise_op<uses_indices, false, indices...>(all_index_seq, new_ix_tup, m, op, is...);
      }
    }
  } // namespace detail


  /**
   * \overload
   * \brief Perform a chipwise nullary operation.
   * \details The nullary operation returns a chip, and that chip is replicated along the specified <code>indices</code>
   * a number of times indicated by <code>Is</code>.
   * \tparam Operation A nullary operation. The operation may optionally take, as arguments,
   * <code>sizeof...(indices)</code> indices (in the same order as <code>indices</code>).
   * \tparam Is Number of dimensions corresponding to each of <code>indices...</code>.
   */
#ifdef __cpp_concepts
  template<std::size_t...indices, typename Operation, value::index...Is> requires (sizeof...(Is) == sizeof...(indices))
#else
  template<std::size_t...indices, typename Operation, typename...Is, std::enable_if_t<
    (value::index<Is> and ...) and (sizeof...(Is) == sizeof...(indices)), int> = 0>
#endif
  constexpr auto
  chipwise_operation(const Operation& operation, Is...is)
  {
    constexpr bool uses_indices = std::is_invocable_v<Operation, std::integral_constant<decltype(indices), 0>...>;

    auto op_result = [](const Operation& operation){
        if constexpr (uses_indices) return operation(std::integral_constant<decltype(indices), 0> {}...);
        else return operation();
      }(operation);
    using OpResult = decltype(op_result);

    static_assert((dimension_size_of_index_is<OpResult, indices, 1, Applicability::permitted> and ...),
      "Operator must return a chip, meaning that the dimension is 1 for each of the specified indices.");
    // Note: set_chip also includes a runtime check that operation() is a chip.

    constexpr std::size_t num_result_indices = std::max({index_count_v<OpResult>, (indices + 1)...});
    auto m = detail::make_nullary_chipwise_default<indices...>(std::make_index_sequence<num_result_indices>{}, op_result, is...);
    set_chip<indices...>(m, std::move(op_result), std::integral_constant<decltype(indices), 0> {}...);

    detail::nullary_chipwise_op<uses_indices, true, indices...>(std::index_sequence<indices...> {}, std::tuple{}, m, operation, is...);
    return m;
  }


} // namespace OpenKalman

#endif //OPENKALMAN_CHIPWISE_OPERATION_HPP
