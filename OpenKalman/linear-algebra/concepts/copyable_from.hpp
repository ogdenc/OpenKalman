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
 * \brief Definition for \ref copyable_to.
 */

#ifndef OPENKALMAN_COPYABLE_FROM_HPP
#define OPENKALMAN_COPYABLE_FROM_HPP

#include "basics/basics.hpp"
#include "linear-algebra/concepts/patterns_may_match_with.hpp"
#include "linear-algebra/traits/element_type_of.hpp"
#include "linear-algebra/traits/get_mdspan.hpp"

namespace OpenKalman
{
#ifndef __cpp_concepts
  namespace detail
  {
    template<typename To, typename From, typename = void>
    struct is_element_copyable : std::false_type {};

    template<typename To, typename From>
    struct is_element_copyable<To, From, std::enable_if_t<
      stdex::assignable_from<typename std::decay_t<decltype(get_mdspan(std::declval<To>()))>::reference, typename element_type_of<From>::type>>>
      : std::true_type {};
  }
#endif


/**
   * \brief Specifies that an \ref indexible object is copyable from another \ref indexible object.
   * \tparam Dest The object to be copied to
   * \tparam Source The object to be copied from
   */
  template<typename Dest, typename Source>
#ifdef __cpp_lib_concepts
  concept copyable_from =
    patterns_may_match_with<Dest, Source> and
    std::assignable_from<typename std::decay_t<decltype(get_mdspan(std::declval<Dest>()))>::reference, element_type_of_t<Source>>;
#else
  constexpr bool copyable_from =
    patterns_may_match_with<Dest, Source> and
    detail::is_element_copyable<Dest, Source>::value;
#endif


}

#endif
