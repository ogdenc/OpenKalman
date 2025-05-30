/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition for \ref reduce function.
 */

#ifndef OPENKALMAN_REDUCE_HPP
#define OPENKALMAN_REDUCE_HPP

namespace OpenKalman
{
  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t index_to_delete, std::size_t...new_indices>
    constexpr auto delete_reduction_index(std::index_sequence<>) { return std::index_sequence<new_indices...>{}; }


    template<std::size_t index_to_delete, std::size_t...new_indices, std::size_t index, std::size_t...indices>
    constexpr auto delete_reduction_index(std::index_sequence<index, indices...>)
    {
      if constexpr (index == index_to_delete)
        return delete_reduction_index<index_to_delete, new_indices...>(std::index_sequence<indices...>{});
      else
        return delete_reduction_index<index_to_delete, new_indices..., index>(std::index_sequence<indices...>{});
    }


    template<typename BinaryOperation, typename Constant, typename Dim>
    constexpr auto constant_reduce_operation(const BinaryOperation& op, const Constant& c, const Dim& dim)
    {
      if constexpr (internal::is_plus<BinaryOperation>::value) return c * dim;
      else if constexpr (internal::is_multiplies<BinaryOperation>::value) return values::pow(c, dim);
      else
      {
        if constexpr (values::dynamic<Dim>)
        {
          if (values::to_number(dim) <= 1) return values::to_number(c);
          else return op(constant_reduce_operation(op, c, values::to_number(dim) - 1), values::to_number(c));
        }
        else if constexpr (Dim::value <= 1) return c;
        else
        {
          auto dim_m1 = std::integral_constant<std::size_t, Dim::value - 1>{};
          return values::operation {op, constant_reduce_operation(op, c, dim_m1), c};
        }
      }
    }


    template<typename BinaryFunction, typename Arg, std::size_t...indices, std::size_t...Ix>
    constexpr decltype(auto)
    reduce_impl(BinaryFunction&& b, Arg&& arg, std::index_sequence<indices...> indices_seq, std::index_sequence<Ix...> seq)
    {
      if constexpr (zero<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
      {
        return make_zero<Arg>(internal::get_reduced_vector_space_descriptor<Ix, indices...>(std::forward<Arg>(arg))...);
      }
      else if constexpr (constant_matrix<Arg>)
      {
        auto dim = internal::count_reduced_dimensions(arg, indices_seq, seq);
        auto c = detail::constant_reduce_operation(b, constant_coefficient{arg}, dim);
        return make_constant<Arg>(std::move(c), internal::get_reduced_vector_space_descriptor<Ix, indices...>(std::forward<Arg>(arg))...);
      }
      else if constexpr (diagonal_matrix<Arg> and internal::is_plus<BinaryFunction>::value and
        not dynamic_dimension<Arg, 0> and not dynamic_dimension<Arg, 1> and
          ((((indices == 1) or ...) and index_dimension_of_v<Arg, 1> >= index_dimension_of_v<Arg, 0>) or
            (((indices == 0) or ...) and ((indices == 1) or ...))))
      {
        return reduce_impl(std::forward<BinaryFunction>(b), diagonal_of(std::forward<Arg>(arg)), delete_reduction_index<1>(indices_seq), seq);
      }
      else if constexpr (diagonal_matrix<Arg> and internal::is_plus<BinaryFunction>::value and
        not dynamic_dimension<Arg, 0> and not dynamic_dimension<Arg, 1> and
        (((indices == 0) or ...) and index_dimension_of_v<Arg, 1> <= index_dimension_of_v<Arg, 0>))
      {
        return reduce_impl(std::forward<BinaryFunction>(b), transpose(diagonal_of(std::forward<Arg>(arg))), delete_reduction_index<0>(indices_seq), seq);
      }
      else if constexpr (diagonal_matrix<Arg> and internal::is_multiplies<BinaryFunction>::value and
        not dynamic_dimension<Arg, 1> and ((indices == 1) or ...))
      {
        if constexpr (index_dimension_of_v<Arg, 1> == 1)
          return reduce_impl(std::forward<BinaryFunction>(b), std::forward<Arg>(arg), delete_reduction_index<0>(indices_seq), seq);
        else
          return reduce_impl(std::forward<BinaryFunction>(b), make_zero(diagonal_of(std::forward<Arg>(arg))), delete_reduction_index<0>(indices_seq), seq);
      }
      else if constexpr (diagonal_matrix<Arg> and internal::is_multiplies<BinaryFunction>::value and
        not dynamic_dimension<Arg, 0> and ((indices == 0) or ...))
      {
        if constexpr (index_dimension_of_v<Arg, 0> == 1)
          return reduce_impl(std::forward<BinaryFunction>(b), std::forward<Arg>(arg), delete_reduction_index<0>(indices_seq), seq);
        else
          return reduce_impl(std::forward<BinaryFunction>(b), make_zero(transpose(diagonal_of(std::forward<Arg>(arg)))), delete_reduction_index<0>(indices_seq), seq);
      }
      else
      {
        using LibraryInterface = interface::library_interface<std::decay_t<Arg>>;
        auto red = LibraryInterface::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
        if constexpr (values::number<decltype(red)>)
          return make_constant<Arg>(std::move(red), internal::get_reduced_vector_space_descriptor<Ix, indices...>(arg)...);
        else
          return red;
      }
    }

  } // namespace detail


  /**
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indices to be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, internal::has_uniform_static_vector_space_descriptors<index, indices...> Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>>
  constexpr indexible decltype(auto)
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
    internal::has_uniform_static_vector_space_descriptors<Arg, index, indices...> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&, typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value, int> = 0>
  constexpr decltype(auto)
#endif
  reduce(BinaryFunction&& b, Arg&& arg)
  {
    if constexpr (dimension_size_of_index_is<Arg, index, 1>) //< Check if Arg is already reduced along index.
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
    }
    else
    {
      return detail::reduce_impl(
        std::forward<BinaryFunction>(b),
        std::forward<Arg>(arg),
        std::index_sequence<index, indices...>{},
        std::make_index_sequence<index_count_v<Arg>>{});
    }
  }


  /**
   * \overload
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, internal::has_uniform_static_vector_space_descriptors Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>>
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<internal::has_uniform_static_vector_space_descriptors<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value, int> = 0>
#endif
  constexpr scalar_type_of_t<Arg>
  reduce(const BinaryFunction& b, const Arg& arg)
  {
    auto seq = std::make_index_sequence<index_count_v<Arg>>{};

    if constexpr (zero<Arg> and (internal::is_plus<BinaryFunction>::value or internal::is_multiplies<BinaryFunction>::value))
    {
      return 0;
    }
    else if constexpr (one_dimensional<Arg>)
    {
      return internal::get_singular_component(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto dim = internal::count_reduced_dimensions(arg, seq, seq);
      return constant_reduce_operation(b, constant_coefficient {arg}, dim);
    }
    else
    {
      decltype(auto) red = detail::reduce_impl(b, arg, seq, seq);
      using Red = decltype(red);

      static_assert(values::number<Red> or one_dimensional<Red, Applicability::permitted>,
        "Incorrect library interface for total 'reduce' on all indices: must return a scalar or one-by-one matrix.");

      if constexpr (values::number<Red>)
        return red;
      else
        return internal::get_singular_component(red);
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_REDUCE_HPP
