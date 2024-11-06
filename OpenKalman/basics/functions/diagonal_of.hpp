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
 * \brief Definition for \ref diagonal_of function.
 */

#ifndef OPENKALMAN_DIAGONAL_OF_HPP
#define OPENKALMAN_DIAGONAL_OF_HPP

namespace OpenKalman
{
  namespace detail
  {
    template<typename T, typename C, typename Indices>
    static constexpr decltype(auto)
    constant_diagonal_of_impl(C&& c, const Indices& indices)
    {
      if constexpr (vector_space_descriptor_tuple<Indices>)
      {      
        auto new_indices = std::tuple_cat(
          std::forward_as_tuple(internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(std::get<0>(indices), std::get<1>(indices))),
          internal::tuple_slice<2, std::tuple_size_v<Indices>>(indices));
        return make_constant<T>(std::forward<C>(c), new_indices);
      }
      else
      {
#if __cpp_lib_containers_ranges >= 202202L and __cpp_lib_ranges_concat >= 202403L
        auto new_indices = std::views::concat(
          internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(std::views::take(indices, 2)), 
          std::views::drop(indices, 2));
#else
        auto it = indices.begin();
        auto new_indices = std::vector<std::iter_value_t<decltype(it)>>{};
        auto i0 = it;
        auto i1 = ++it;
        if (i0 == indices.end())
          new_indices.emplace_back(0);
        else if (i1 == indices.end())
          new_indices.emplace_back(1);
        else
        {
          auto d0 = internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(*i0, *i1);
          auto new_indices = std::vector {{d0}};
          std::copy(++it, indices.end(), ++new_indices.begin());
        }
#endif
        return make_constant<T>(std::forward<C>(c), new_indices);
      }
    }

  } // namespace detail


  /**
   * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
   * \tparam Arg An \ref indexible object, which can have any rank and may or may not be square
   * \returns Arg A column vector whose \ref vector_space_descriptor corresponds to the smallest-dimension index.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires (index_count_v<Arg> == dynamic_size) or (index_count_v<Arg> <= 2)
  constexpr indexible decltype(auto)
#else
  template<typename Arg, std::enable_if_t<(index_count_v<Arg> == dynamic_size or index_count_v<Arg> <= 2), int> = 0>
  constexpr decltype(auto)
#endif
  diagonal_of(Arg&& arg)
  {
    if constexpr (diagonal_adapter<Arg>)
    {
      return nested_object(std::forward<Arg>(arg));
    }
    else if constexpr (diagonal_adapter<Arg, 1>)
    {
      return transpose(nested_object(std::forward<Arg>(arg)));
    }
    else if constexpr (one_dimensional<Arg>)
    {
      return std::forward<Arg>(arg);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      auto ds = all_vector_space_descriptors(std::forward<Arg>(arg));
      if constexpr (vector_space_descriptor_tuple<decltype(ds)>)
      {      
        return detail::constant_diagonal_of_impl<Arg>(
          constant_coefficient {std::forward<Arg>(arg)},
          std::tuple_cat(ds, std::tuple{Dimensions<1>{}, Dimensions<1>{}}));
      }
      else
      {
        return detail::constant_diagonal_of_impl<Arg>(constant_coefficient {std::forward<Arg>(arg)}, ds);
      }
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      auto ds = all_vector_space_descriptors(std::forward<Arg>(arg));
      if constexpr (vector_space_descriptor_tuple<decltype(ds)>)
      {      
        return detail::constant_diagonal_of_impl<Arg>(
          constant_diagonal_coefficient {std::forward<Arg>(arg)},
          std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{Dimensions<1>{}, Dimensions<1>{}}));
      }
      else
      {
        return detail::constant_diagonal_of_impl<Arg>(constant_diagonal_coefficient {std::forward<Arg>(arg)}, ds);
      }
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
