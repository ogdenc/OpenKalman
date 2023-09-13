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

#ifndef OPENKALMAN_CHIPWISE_OPERATIONS_HPP
#define OPENKALMAN_CHIPWISE_OPERATIONS_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<std::size_t ix, typename Best_d>
    const Best_d& chipwise_descriptor_for(const Best_d& best_d) { return best_d; }

    template<std::size_t ix, typename Best_d, typename Arg, typename...Args>
    decltype(auto) chipwise_descriptor_for(const Best_d& best_d, const Arg& arg, const Args&...args)
    {
      auto d = get_index_descriptor<ix>(arg);
      using D = decltype(d);
      if constexpr (fixed_index_descriptor<Best_d>)
      {
        if constexpr (fixed_index_descriptor<D>)
          static_assert(dimension_size_of_v<D> == dimension_size_of_v<Best_d>,
            "Arguments to chipwise_operation must have matching index descriptors.");
        else
          if (d != best_d) throw std::invalid_argument {"Arguments to chipwise_operation must have matching index descriptors."};
        return chipwise_descriptor_for<ix>(best_d, args...);
      }
      else // dynamic_index_descriptor<Best_d>
      {
        if (d != best_d) throw std::invalid_argument {"Arguments to chipwise_operation must have matching index descriptors."};
        if constexpr (fixed_index_descriptor<D>)
          return chipwise_descriptor_for<ix>(d, args...);
        else
          return chipwise_descriptor_for<ix>(best_d, args...);
      }
    }

    template<std::size_t...ix, typename Arg, typename...Args>
    auto make_chipwise_default(std::index_sequence<ix...>, const Arg& arg, const Args&...args)
    {
      return make_default_dense_writable_matrix_like<Arg>(chipwise_descriptor_for<ix>(get_index_descriptor<ix>(arg), args...)...);
    }


    template<bool uses_indices, std::size_t...indices,
      std::size_t...index_ixs, typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    chipwise_op_chip(std::index_sequence<indices...>, std::index_sequence<index_ixs...>, const Ix_tup& ix_tup,
      M& m, const Op& op, Args&&...args)
    {
      if constexpr (uses_indices)
      {
        auto chip = op(get_chip<indices...>(std::forward<Args>(args), std::get<index_ixs>(ix_tup)...)..., std::get<index_ixs>(ix_tup)...);
        set_chip<indices...>(m, std::move(chip), std::get<index_ixs>(ix_tup)...);
      }
      else
      {
        auto chip = op(get_chip<indices...>(std::forward<Args>(args), std::get<index_ixs>(ix_tup)...)...);
        set_chip<indices...>(m, std::move(chip), std::get<index_ixs>(ix_tup)...);
      }
    }

    template<bool uses_indices, typename Indices_seq, typename Ix_tup, typename M, typename Op, typename...Args>
    constexpr void
    chipwise_op(Indices_seq indices_seq, const Ix_tup& ix_tup, M& m, const Op& op, Args&&...args)
    {
      std::make_index_sequence<std::tuple_size_v<Ix_tup>> index_ix_seq;
      chipwise_op_chip<uses_indices>(indices_seq, index_ix_seq, ix_tup, m, op, std::forward<Args>(args)...);
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
   * \brief Perform a chipwise n-ary operation on one or more indexible arguments.
   * \details Given an indexible object of rank <var>n</var>, a "chip" is a subset of that object, having rank in
   * the range (0, <var>n</var>]. This function takes same-size chips from each of the arguments (if any) and
   * performs an operation returning a chip (of the same size), for every possible chip within the result.
   * \tparam indices The reduced-dimension indices which will be replicated to fill the result
   * \tparam Operation An operation (unary, binary, etc.). In addition to taking one or more chips as arguments,
   * the operation may also take <code>sizeof...(indices)</code> indices (in the same order as <code>indices</code>).
   * \tparam Args The arguments, which must be the same size.
   * \result An object of the same size as the arguments (if any) or
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
      std::make_index_sequence<std::max({max_indices_of_v<Args>...})> args_ix_seq;
      auto m = detail::make_chipwise_default(args_ix_seq, args...);

      constexpr bool uses_indices = std::is_invocable_v<Operation,
        decltype(get_chip<indices...>(std::declval<Args>(), std::integral_constant<decltype(indices), 0>{}...))...,
        std::integral_constant<decltype(indices), 0>...>;

      std::index_sequence<indices...> indices_seq;
      detail::chipwise_op<uses_indices, indices...>(indices_seq, std::tuple{}, m, operation, std::forward<Args>(args)...);
      return m;
    }
    else
      return make_self_contained<Args...>(operation(std::forward<Args>(args)...));
  }


  namespace detail
  {
    template<std::size_t op_ix, typename Ds_tup>
    auto nullary_chipwise_descriptor(const Ds_tup& ds_tup)
    {
      return std::get<op_ix>(ds_tup);
    }

    template<std::size_t op_ix, std::size_t index, std::size_t...indices, typename Ds_tup, typename I, typename...Is>
    auto nullary_chipwise_descriptor(const Ds_tup& ds_tup, I i, Is...is)
    {
      if constexpr (op_ix == index) return internal::replicate_index_descriptor(std::get<op_ix>(ds_tup), i);
      else return nullary_chipwise_descriptor<op_ix, indices...>(ds_tup, is...);
    }

    template<std::size_t...indices, std::size_t...op_ix, typename OpResult, typename...Is>
    auto make_nullary_chipwise_default(std::index_sequence<op_ix...>, const OpResult& op_result, Is...is)
    {
      auto ds_tup = get_all_dimensions_of(op_result);
      return make_default_dense_writable_matrix_like<OpResult>(nullary_chipwise_descriptor<op_ix, indices...>(ds_tup, is...)...);
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
  template<std::size_t...indices, typename Operation, index_value...Is> requires (sizeof...(Is) == sizeof...(indices))
#else
  template<std::size_t...indices, typename Operation, typename...Is, std::enable_if_t<
    (index_value<Is> and ...) and (sizeof...(Is) == sizeof...(indices)), int> = 0>
#endif
  constexpr auto
  chipwise_operation(const Operation& operation, Is...is)
  {
    constexpr bool uses_indices = std::is_invocable_v<Operation, std::integral_constant<decltype(indices), 0>...>;

    auto op_result = [](const Operation& operation){
        if constexpr (uses_indices) return make_self_contained(operation(std::integral_constant<decltype(indices), 0> {}...));
        else return make_self_contained(operation());
      }(operation);
    using OpResult = decltype(op_result);

    static_assert((dimension_size_of_index_is<OpResult, indices, 1, Likelihood::maybe> and ...),
      "Operator must return a chip, meaning that the dimension is 1 for each of the specified indices.");
    // Note: set_chip includes a runtime check that operation() is a chip.

    std::make_index_sequence<max_indices_of_v<OpResult>> op_ix_seq;
    auto m = detail::make_nullary_chipwise_default<indices...>(op_ix_seq, op_result, is...);
    set_chip<indices...>(m, std::move(op_result), std::integral_constant<decltype(indices), 0> {}...);

    detail::nullary_chipwise_op<uses_indices, true, indices...>(std::index_sequence<indices...> {}, std::tuple{}, m, operation, is...);
    return m;
  }


} // namespace OpenKalman

#endif //OPENKALMAN_CHIPWISE_OPERATIONS_HPP
