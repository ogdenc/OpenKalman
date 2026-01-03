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
 * \brief Definitions for \ref make_constant_diagonal.
 */

#ifndef OPENKALMAN_MAKE_CONSTANT_DIAGONAL_HPP
#define OPENKALMAN_MAKE_CONSTANT_DIAGONAL_HPP

#include "patterns/patterns.hpp"
#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "make_constant.hpp"
#include "to_diagonal.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every diagonal element is a constant \ref values::value "value".
   * \returns An mdspan returning the constant value at every set of indices.
   * \param c A \ref values::value "value"
   * \param p a \ref patterns::pattern_collection for the result (e.g., an std::extents object).
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern_collection P>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    values::value<C> and patterns::pattern_collection<P>, int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c, P&& p)
  {
    auto nested_p = patterns::pattern_collection_of_diagonal(patterns::to_extents(p));
    return to_diagonal(make_constant(std::move(c), std::move(nested_p)), std::forward<P>(p));
  }


  /**
   * \overload
   * \brief The \ref patterns::pattern_collection "pattern_collection" is constructed from a list of \ref patterns::patterns "patterns".
   */
#ifdef __cpp_concepts
  template<values::value C, patterns::pattern...Ps>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename...Ps, std::enable_if_t<values::value<C> and (... and patterns::pattern<Ps>), int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c, Ps&&...ps)
  {
    return make_constant_diagonal(std::move(c), std::tuple{std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_diagonal_object based on a default-initializable \ref patterns::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<patterns::pattern_collection P, values::value C> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    patterns::pattern_collection<P> and
    values::value<C> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c)
  {
    return make_constant_diagonal(std::move(c), P{});
  }


}

#endif
