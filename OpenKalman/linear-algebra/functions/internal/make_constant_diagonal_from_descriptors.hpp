/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2024 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \internal
 * \brief Definition for \ref make_constant_diagonal_from_descriptors.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_DIAGONAL_FROM_DESCRIPTORS_HPP
#define OPENKALMAN_MAKE_CONSTANT_DIAGONAL_FROM_DESCRIPTORS_HPP

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Extract a column vector (or column slice for rank>2 tensors) comprising the diagonal elements.
   * \tparam Arg An \ref indexible object, which can have any rank and may or may not be square
   * \returns Arg A column vector whose \ref vector_space_descriptor corresponds to the smallest-dimension index.
   */
  template<typename T, typename C, typename Descriptors>
  static constexpr decltype(auto)
  make_constant_diagonal_from_descriptors(C&& c, const Descriptors& descriptors)
  {
    if constexpr (vector_space_descriptor_tuple<Descriptors>)
    {
      auto new_descriptors = std::tuple_cat(
        std::forward_as_tuple(internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(std::get<0>(descriptors), std::get<1>(descriptors))),
        internal::tuple_slice<2, std::tuple_size_v<Descriptors>>(descriptors));
      return make_constant<T>(std::forward<C>(c), new_descriptors);
    }
    else
    {
#if __cpp_lib_containers_ranges >= 202202L and __cpp_lib_ranges_concat >= 202403L
      auto new_indices = std::views::concat(
        internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(std::ranges::views::take(indices, 2)),
        indices | std::ranges::views::drop(2));
#else
      auto it = descriptors.begin();
      auto new_descriptors = std::vector<std::decay_t<decltype(*it)>>{};
      auto i0 = it;
      auto i1 = ++it;
      if (i1 == descriptors.end())
      {
        new_descriptors.emplace_back(descriptors::Axis{});
      }
      else if (i0 != descriptors.end())
      {
        auto d0 = internal::smallest_vector_space_descriptor<scalar_type_of_t<T>>(*i0, *i1);
        new_descriptors.emplace_back(d0);
        std::copy(++it, descriptors.end(), ++new_descriptors.begin());
      }
#endif
      return make_constant<T>(std::forward<C>(c), new_descriptors);
    }
  }

} // namespace OpenKalman

#endif //OPENKALMAN_MAKE_CONSTANT_DIAGONAL_FROM_DESCRIPTORS_HPP
