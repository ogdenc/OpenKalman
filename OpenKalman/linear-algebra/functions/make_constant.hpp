/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022-2026 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Definitions for \ref make_constant.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_HPP
#define OPENKALMAN_MAKE_CONSTANT_HPP

#include "linear-algebra/concepts/constant_object.hpp"
#include "linear-algebra/functions/attach_patterns.hpp"
#include "linear-algebra/interfaces/stl/layout_constant.hpp"
#include "linear-algebra/interfaces/stl/constant_accessor.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every element is a constant \ref values::value "value".
   * \returns An mdspan returning the constant value at every set of indices.
   * \param c A \ref values::value "value"
   * \param p a \ref patterns::pattern_collection (e.g., an std::extents object).
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P> requires values::fixed<collections::size_of<P>>
  constexpr constant_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    values::value<C> and
    patterns::pattern_collection<P> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant(C c, P&& p)
  {
    decltype(auto) extents = patterns::to_extents(std::forward<P>(p));
    typename interface::layout_constant::mapping<std::decay_t<decltype(extents)>> mapping(extents);
    interface::constant_accessor<C> accessor;
    auto m = stdex::mdspan {std::move(c), mapping, accessor};
    return attach_patterns(std::move(m), std::forward<P>(p));
  }


  /**
   * \overload
   * \brief The \ref patterns::pattern_collection "pattern_collection" is constructed from a list of \ref patterns::patterns "patterns".
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern...Ps>
  constexpr constant_object auto
#else
  template<typename C, typename...Ps, std::enable_if_t<values::value<C> and (... and patterns::pattern<Ps>), int> = 0>
  constexpr auto
#endif
  make_constant(C c, Ps&&...ps)
  {
    return make_constant(std::move(c), std::tuple{std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_object based on a default-initializable \ref patterns::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<patterns::pattern_collection P, values::value C> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr constant_object auto
#else
  template<typename P, typename C, std::enable_if_t<
    patterns::pattern_collection<P> and
    values::value<C> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant(C c)
  {
    return make_constant(std::move(c), P{});
  }


  namespace interface
  {
    template<typename N, typename E>
    struct object_traits<stdex::mdspan<N, E, layout_constant, constant_accessor<N>>>
    {
      static const bool is_specialized = true;

      template<typename M>
      static constexpr decltype(auto)
      get_mdspan(M&& m) { return std::forward<M>(m); }

      template<typename M>
      static constexpr auto
      get_constant(M&& m)
      {
        return m.accessor().access(m.data_handle(), 0);
      }

    };
  };

}

#endif
