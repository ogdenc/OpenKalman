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

#include "linear-algebra/concepts/constant_diagonal_object.hpp"
#include "to_diagonal.hpp"
#include "make_constant.hpp"

namespace OpenKalman
{
  /**
   * \brief Make an \ref indexible object in which every diagonal element is a constant \ref values::value "value".
   * \returns An mdspan returning the constant value at every set of indices.
   * \param c A \ref values::value "value"
   * \param extents An std::extents object defining the extents.
   */
#ifdef __cpp_concepts
  template<values::value C, typename IndexType, std::size_t...Extents>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename IndexType, std::size_t...Extents, std::enable_if_t<values::value<C>, int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c, stdex::extents<IndexType, Extents...> extents)
  {
    return to_diagonal(make_constant(std::move(c), std::move(extents)));
  }



  /**
   * \overload
   * \brief \ref Make a \ref constant_diagonal_object based on a \ref coordinates::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<values::value C, coordinates::pattern_collection P> requires values::fixed<collections::size_of<P>>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    values::value<C> and
    coordinates::pattern_collection<P> and
    values::fixed<collections::size_of<P>>, int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c, P&& p)
  {
    return to_diagonal(make_constant(std::move(c), std::forward<P>(p)));
  }


  /**
   * \overload
   * \brief The \ref coordinates::pattern_collection "pattern_collection" is constructed from a list of \ref coordinates::patterns "patterns".
   */
#ifdef __cpp_concepts
  template<values::value C, coordinates::pattern...Ps>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename...Ps, std::enable_if_t<values::value<C> and (... and coordinates::pattern<Ps>), int> = 0>
  constexpr auto
#endif
  make_constant_diagonal(C c, Ps&&...ps)
  {
    return make_constant_diagonal(std::move(c), std::tuple{std::forward<Ps>(ps)...});
  }


  /**
   * \overload
   * \brief \ref Make a \ref constant_diagonal_object based on a default-initializable \ref coordinates::pattern_collection "pattern_collection".
   */
#ifdef __cpp_concepts
  template<coordinates::pattern_collection P, values::value C> requires
    std::default_initializable<P> and
    values::fixed<collections::size_of<P>>
  constexpr constant_diagonal_object auto
#else
  template<typename C, typename P, std::enable_if_t<
    coordinates::pattern_collection<P> and
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
