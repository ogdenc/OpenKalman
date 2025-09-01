/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2025 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definition of \ref coordinates::repeat_tuple_view and \ref coordinates::views::repeat.
 */

#ifndef OPENKALMAN_COORDINATES_VIEWS_REPEAT_HPP
#define OPENKALMAN_COORDINATES_VIEWS_REPEAT_HPP

#include "collections/collections.hpp"

namespace OpenKalman::coordinates::views
{
  namespace detail
  {
    struct repeat_adaptor
    {
#ifdef __cpp_lib_ranges
      template<std::move_constructible W, values::size Bound = stdcompat::unreachable_sentinel_t> requires
        (OpenKalman::internal::is_signed_integer_like<values::value_type_of_t<Bound>> or
        (OpenKalman::internal::is_integer_like<values::value_type_of_t<Bound>> and stdcompat::weakly_incrementable<values::value_type_of_t<Bound>> or
        std::same_as<Bound, std::unreachable_sentinel_t>))
#else
      template<typename W, typename Bound = stdcompat::unreachable_sentinel_t, typename = void>
#endif
      constexpr auto
      operator() [[nodiscard]] (W&& value, Bound&& bound = {}) const
      {
        if constexpr (std::is_same_v<Bound, stdcompat::unreachable_sentinel_t>)
          return stdcompat::ranges::views::repeat(std::forward<W>(value)) | all;
        else if constexpr (values::fixed<Bound>)
        {
          if constexpr (values::fixed_value_of_v<Bound> == 1)
            return stdcompat::ranges::views::single(std::forward<W>(value)) | all;
          else
            return repeat_tuple_view<values::fixed_value_of_v<Bound>, W> {std::forward<W>(value)} | all;
        }
        else
          return stdcompat::ranges::views::repeat(std::forward<W>(value), values::to_value_type(std::forward<Bound>(bound))) | all;
      }
    };

  }


  /**
   * \brief a std::ranges::range_adaptor_closure for a set of repeated \ref descriptor objects.
   */
  inline constexpr detail::repeat_adaptor repeat;

}


#endif 
