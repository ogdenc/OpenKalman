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
 * \brief Definitions for \ref make_zero.
 */

#ifndef OPENKALMAN_MAKE_ZERO_HPP
#define OPENKALMAN_MAKE_ZERO_HPP

#include "make_constant.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every element is 0.
   * \returns An mdspan returning the constant value at every set of indices.
   * \tparam C The value type
   * \param p a \ref patterns::pattern_collection (e.g., and std::extents object).
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P> requires values::fixed<collections::size_of<P>>
  constexpr zero auto
#else
  template<typename C, typename P, std::enable_if_t<values::value<C> and
    patterns::pattern_collection<P> and values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_zero(P&& p)
  {
    return make_constant(values::fixed_value<C, 0>{}, std::forward<P>(p));
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
  make_zero(Ps&&...ps)
    {
      return make_zero<C>(std::tuple{std::forward<Ps>(ps)...});
    }


  /**
   * \overload
   * \brief \ref Make a \ref zero based on a default-initializable \ref patterns::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr constant_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    patterns::pattern_collection<P> and
    values::value<C> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_zero()
  {
    return make_zero<C>(P{});
  }



}

#endif
