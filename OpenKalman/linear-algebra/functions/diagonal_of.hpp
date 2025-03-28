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
  /**
   * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
   * \tparam Arg An \ref indexible object, which can have any rank and may or may not be square
   * \returns Arg A column vector whose \ref coordinate::pattern corresponds to the smallest-dimension index.
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
      if constexpr (pattern_tuple<decltype(ds)>)
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(
          constant_coefficient {std::forward<Arg>(arg)},
          std::tuple_cat(ds, std::tuple{coordinate::Axis{}, coordinate::Axis{}}));
      }
      else
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(constant_coefficient {std::forward<Arg>(arg)}, ds);
      }
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      auto ds = all_vector_space_descriptors(std::forward<Arg>(arg));
      if constexpr (pattern_tuple<decltype(ds)>)
      {      
        return internal::make_constant_diagonal_from_descriptors<Arg>(
          constant_diagonal_coefficient {std::forward<Arg>(arg)},
          std::tuple_cat(all_vector_space_descriptors(std::forward<Arg>(arg)), std::tuple{coordinate::Axis{}, coordinate::Axis{}}));
      }
      else
      {
        return internal::make_constant_diagonal_from_descriptors<Arg>(constant_diagonal_coefficient {std::forward<Arg>(arg)}, ds);
      }
    }
    else
    {
      return interface::library_interface<std::decay_t<Arg>>::diagonal_of(std::forward<Arg>(arg));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_DIAGONAL_OF_HPP
