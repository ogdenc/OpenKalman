/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2023 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief definitions for \ref to_euclidean function.
 */

#ifndef OPENKALMAN_TO_EUCLIDEAN_HPP
#define OPENKALMAN_TO_EUCLIDEAN_HPP

namespace OpenKalman
{
  /**
   * \brief Transform a matrix or tensor into Euclidean space along its first index.
   * \tparam Arg A matrix or tensor. I
   */
#ifdef __cpp_concepts
  template<wrappable Arg, index_descriptor C>
  requires dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
    equivalent_to<C, index_descriptor_of_t<Arg, 0>>
#else
  template<typename Arg, typename C, std::enable_if_t<wrappable<Arg> and index_descriptor<C> and
    (dynamic_index_descriptor<C> or dynamic_rows<Arg> or has_untyped_index<Arg, 0> or
      equivalent_to<C, index_descriptor_of_t<Arg, 0>>), int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg, const C& c) noexcept
  {
    if constexpr (dynamic_dimension<Arg, 0> and not euclidean_index_descriptor<index_descriptor_of_t<Arg, 0>>)
      if (not get_index_descriptor_is_euclidean(get_index_descriptor<0>(arg)) and c != get_index_descriptor<0>(arg))
        throw std::domain_error {"In to_euclidean, specified index descriptor does not match that of the object's index 0"};

    if constexpr (euclidean_index_descriptor<C>)
    {
      return std::forward<Arg>(arg);
    }
    else
    {
      if constexpr (has_dynamic_dimensions<Arg>) if (not get_wrappable(arg))
        throw std::domain_error {"Argument of to_euclidean is not wrappable"};

      return interface::ModularTransformationTraits<Arg>::to_euclidean(std::forward<Arg>(arg), c);
    }
  }


#ifdef __cpp_concepts
  template<all_fixed_indices_are_euclidean Arg>
#else
  template<typename Arg, std::enable_if_t<all_fixed_indices_are_euclidean<Arg>, int> = 0>
#endif
  constexpr decltype(auto)
  to_euclidean(Arg&& arg)
  {
    return to_euclidean(std::forward<Arg>(arg), get_index_descriptor<0>(arg));
  }


} // namespace OpenKalman

#endif //OPENKALMAN_TO_EUCLIDEAN_HPP
