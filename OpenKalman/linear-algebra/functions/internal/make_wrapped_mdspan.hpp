/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \internal
 * \file
 * \brief Definition for \ref internal::make_wrapped_mdspan.
 */

#ifndef OPENKALMAN_MAKE_WRAPPED_MDSPAN_HPP
#define OPENKALMAN_MAKE_WRAPPED_MDSPAN_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/indexible.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/adapters/internal/owning_array.hpp"

namespace OpenKalman::internal
{
  namespace detail
  {
    template<class T>
    struct is_mdspan : std::false_type {};

    template<class ElementType, class Extents, class Layout, class Accessor>
    struct is_mdspan<stdex::mdspan<ElementType, Extents, Layout, Accessor>> : std::true_type {};
  }


  /**
   * \internal
   * \brief Make an mdspan, potentially wrapping it in an \ref owning_array and attaching patterns.
   * \details Any pattern_collection already associated with the argument will be overwritten.
   */
#ifdef __cpp_concepts
  template<indexible Arg, typename F, typename M, typename A, patterns::pattern_collection P> requires
    std::is_invocable_r_v<typename std::decay_t<A>::data_handle_type, F,
      typename std::decay_t<decltype(get_mdspan(std::declval<Arg>()))>::data_handle_type>
#else
  template<typename Arg, typename F, typename M, typename A, typename P, std::enable_if_t<
    indexible<Arg> and patterns::pattern_collection<P> and
    std::is_invocable_r_v<typename std::decay_t<A>::data_handle_type, F,
      typename std::decay_t<decltype(get_mdspan(std::declval<Arg>()))>::data_handle_type>, int> = 0>
#endif
  constexpr auto
  make_wrapped_mdspan(Arg&& arg, F&& f, M&& m, A&& a, P&& p)
  {
    decltype(auto) darg = detach_patterns(std::forward<Arg>(arg));
    using DArg = decltype(darg);
    if constexpr (detail::is_mdspan<std::decay_t<DArg>>::value or std::is_lvalue_reference_v<DArg>)
    {
      return attach_patterns(
        stdex::mdspan(
          f(get_mdspan(std::forward<DArg>(darg)).data_handle()),
          std::forward<M>(m),
          std::forward<A>(a)),
        std::forward<P>(p));
    }
    else
    {
      return attach_patterns(
        owning_array {
          std::forward<DArg>(darg),
          std::forward<F>(f),
          std::forward<M>(m),
          std::forward<A>(a)},
        std::forward<P>(p));
    }
  }

}

#endif
