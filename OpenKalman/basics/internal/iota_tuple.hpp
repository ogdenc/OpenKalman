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
 * \internal
 * \brief Definition for \ref internal::iota_tuple.
 */

#ifndef OPENKALMAN_IOTA_TUPLE_HPP
#define OPENKALMAN_IOTA_TUPLE_HPP

#include <type_traits>
#include "collection_size_of.hpp"

namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief A \ref internal::tuple_like iota
   * \details A tuple-like sequence in the form of
   * <code>std::integral_sequence<std::size_t, 0>{},...,std::integral_sequence<std::size_t, N>{}</code>
   */
  template<std::size_t start, std::size_t bound>
  struct IotaTuple {};


  template<size_t i, std::size_t start, std::size_t bound>
  constexpr auto
  get(const IotaTuple<start, bound>&)
  {
    static_assert(start + i < bound);
    return std::integral_constant<size_t, start + i>{};
  }

} // OpenKalman::internal

namespace std
{
  template<std::size_t start, std::size_t bound>
  struct tuple_size<OpenKalman::internal::IotaTuple<start, bound>> : std::integral_constant<size_t, bound - start> {};


  template<std::size_t i, std::size_t start, std::size_t bound>
  struct tuple_element<i, OpenKalman::internal::IotaTuple<start, bound>>
  {
    static_assert(start + i < bound);
    using type = std::integral_constant<size_t, start + i>;
  };

} // namespace std


namespace OpenKalman::internal
{
  /**
   * \internal
   * \brief Create a \ref internal::tuple_like iota
   * \details The result is a tuple-like sequence in the form of
   * <code>std::integral_sequence<std::size_t, 0>{},...,std::integral_sequence<std::size_t, N>{}</code>
   */
  template<std::size_t start, std::size_t bound>
  constexpr auto
  iota_tuple()
  {
    return IotaTuple<start, bound>{};
  };

} // OpenKalman::internal


#endif //OPENKALMAN_IOTA_TUPLE_HPP
